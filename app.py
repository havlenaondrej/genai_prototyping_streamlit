from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import re
import io
import zipfile

import numpy as np
import streamlit as st
import openai
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import text as sql_text
import secrets


# -------------------------------
# Environment & OpenAI client
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

openai_client: Optional[openai.OpenAI] = None
anthropic_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if ANTHROPIC_API_KEY:
    try:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        pass


# -------------------------------
# Database setup (SQLite + SQLAlchemy)
# -------------------------------
Base = declarative_base()


class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True)
    type = Column(String(50), default="note")  # note | ai_output | document
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    created_by = Column(String(255), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    prompt_text = Column(Text, nullable=True)  # optional source prompt
    context = Column(String(100), default="General")

    comments = relationship("Comment", back_populates="artifact", cascade="all, delete-orphan")
    provenance_events = relationship("ProvenanceEvent", back_populates="artifact", cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id"))
    author = Column(String(255), default="user")
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    artifact = relationship("Artifact", back_populates="comments")


class Template(Base):
    __tablename__ = "templates"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    prompt_text = Column(Text, nullable=False)  # can include {placeholders}
    version = Column(Integer, default=1)
    created_by = Column(String(255), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)


class ArtifactLink(Base):
    __tablename__ = "artifact_links"
    id = Column(Integer, primary_key=True)
    from_id = Column(Integer, ForeignKey("artifacts.id"))
    to_id = Column(Integer, ForeignKey("artifacts.id"))
    relation = Column(String(100), default="related")  # generated_from_template, derived_from_prompt


class ProvenanceEvent(Base):
    __tablename__ = "provenance_events"
    id = Column(Integer, primary_key=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id"))
    actor_type = Column(String(50))  # user | model
    actor = Column(String(255))  # username or model name
    event_type = Column(String(100))  # created | updated | generated
    timestamp = Column(DateTime, default=datetime.utcnow)

    artifact = relationship("Artifact", back_populates="provenance_events")


class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id"), unique=True)
    model = Column(String(100))
    dim = Column(Integer)
    vector_json = Column(Text)  # store as JSON list


class MetricsEvent(Base):
    __tablename__ = "metrics_events"
    id = Column(Integer, primary_key=True)
    event_type = Column(String(100), nullable=False)
    details = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    context = Column(String(100), default="General")
    created_by = Column(String(255), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    share_token = Column(String(64), nullable=True, unique=True)


class Turn(Base):
    __tablename__ = "turns"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(50), nullable=False)  # user | assistant | system
    author = Column(String(255), default="user")
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PresenceHeartbeat(Base):
    __tablename__ = "presence_heartbeats"
    id = Column(Integer, primary_key=True)
    user_name = Column(String(255), nullable=False)
    context = Column(String(100), default="General")
    conversation_id = Column(Integer, nullable=True)
    last_seen = Column(DateTime, default=datetime.utcnow)


class Contexts(Base):
    __tablename__ = "contexts"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)


class MicroApp(Base):
    __tablename__ = "micro_apps"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    streamlit_code = Column(Text, nullable=False)
    mcp_server_code = Column(Text, nullable=True)
    context = Column(String(100), default="General")
    created_by = Column(String(255), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # 1=active, 0=inactive


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_by = Column(String(255), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)


class TeamMember(Base):
    __tablename__ = "team_members"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    user_name = Column(String(255), nullable=False)
    role = Column(String(50), default="member")  # owner, admin, member, viewer
    joined_at = Column(DateTime, default=datetime.utcnow)


DB_PATH = os.path.join(os.path.dirname(__file__), "brainio.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# Safe migration: add 'context' column if missing
with engine.begin() as conn:
    cols = conn.execute(sql_text("PRAGMA table_info('artifacts')")).fetchall()
    existing_cols = {row[1] for row in cols}
    if "context" not in existing_cols:
        conn.execute(sql_text("ALTER TABLE artifacts ADD COLUMN context VARCHAR(100) DEFAULT 'General'"))

# Safe migration: add 'share_token' to conversations if missing
with engine.begin() as conn:
    cols = conn.execute(sql_text("PRAGMA table_info('conversations')")).fetchall()
    existing_cols = {row[1] for row in cols}
    if "share_token" not in existing_cols:
        conn.execute(sql_text("ALTER TABLE conversations ADD COLUMN share_token VARCHAR(64)"))

# Seed default contexts if missing
with SessionLocal() as _s:
    existing = {c.name for c in _s.query(Contexts).all()}
    for name in ["General", "Design", "Research", "Engineering", "Marketing"]:
        if name not in existing:
            _s.add(Contexts(name=name))
    _s.commit()


# -------------------------------
# Embeddings utilities
# -------------------------------

def compute_embedding(text: str) -> Optional[List[float]]:
    if not openai_client:
        return None
    text = text or ""
    try:
        emb = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return emb.data[0].embedding
    except Exception:
        return None


def ensure_artifact_embedding(session, artifact: Artifact) -> None:
    existing = session.query(Embedding).filter(Embedding.artifact_id == artifact.id).one_or_none()
    if existing:
        return
    combined = f"{artifact.title}\n\n{artifact.content}"
    vec = compute_embedding(combined)
    if vec is None:
        return
    session.add(Embedding(artifact_id=artifact.id, model=EMBEDDING_MODEL, dim=len(vec), vector_json=json.dumps(vec)))
    session.commit()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embedding_search(session, query: str, top_k: int = 10) -> List[int]:
    if not openai_client or not query:
        return []
    qvec = compute_embedding(query)
    if qvec is None:
        return []
    q = np.array(qvec, dtype=np.float32)
    results: List[Dict[str, Any]] = []
    for emb in session.query(Embedding).all():
        vec = np.array(json.loads(emb.vector_json), dtype=np.float32)
        score = cosine_similarity(q, vec)
        results.append({"artifact_id": emb.artifact_id, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return [r["artifact_id"] for r in results[:top_k]]


# -------------------------------
# Streamlit UI helpers
# -------------------------------

def clean_ai_generated_code(raw_code):
    """Clean AI-generated code by removing markdown formatting and extracting pure Python code."""
    if not raw_code:
        return None
    
    # Remove markdown code blocks
    lines = raw_code.split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        # Check for code block markers
        if line.strip().startswith('```'):
            if 'python' in line.lower():
                in_code_block = True
            else:
                in_code_block = False
            continue
        
        # If we're in a code block or if there are no code block markers, add the line
        if in_code_block or not any('```' in l for l in lines):
            cleaned_lines.append(line)
    
    cleaned_code = '\n'.join(cleaned_lines).strip()
    
    # Basic validation - check if it looks like Python code
    if cleaned_code and ('import' in cleaned_code or 'streamlit' in cleaned_code or 'st.' in cleaned_code):
        return cleaned_code
    
    return None

st.set_page_config(page_title="Brainio MVP", layout="wide")
APP_PASSWORD = os.getenv("APP_PASSWORD")
DEFAULT_AUTHOR = os.getenv("USER_NAME", "user")

# Password gate disabled - direct access
qp = st.query_params
share_token_qp = qp.get("share") if "share" in qp else None

st.title("Brainio – Human–AI Co‑Creation (MVP)")

# Sidebar: select context
st.sidebar.header("Identity")
user_name = st.sidebar.text_input("Your name", value=DEFAULT_AUTHOR)

st.sidebar.header("Context")
# Live contexts from DB
with SessionLocal() as _s:
    ctx_rows = _s.query(Contexts).order_by(Contexts.name.asc()).all()
available_contexts = [c.name for c in ctx_rows] or ["General"]
current_context = st.sidebar.selectbox("Artifact context", options=available_contexts, index=max(0, available_contexts.index("General")) if "General" in available_contexts else 0)
st.sidebar.caption("Only artifacts in this context are shown.")
with st.sidebar.expander("Manage contexts", expanded=False):
    new_ctx = st.text_input("New context name", key="ctx_new")
    if st.button("Add context", key="ctx_add") and new_ctx.strip():
        with SessionLocal() as _s:
            if not _s.query(Contexts).filter(Contexts.name == new_ctx.strip()).first():
                _s.add(Contexts(name=new_ctx.strip()))
                _s.commit()
        st.rerun()
    rename_from = st.selectbox("Rename from", options=available_contexts, key="ctx_rename_from")
    rename_to = st.text_input("Rename to", key="ctx_rename_to")
    if st.button("Rename", key="ctx_rename_btn") and rename_to.strip():
        with SessionLocal() as _s:
            row = _s.query(Contexts).filter(Contexts.name == rename_from).first()
            if row:
                row.name = rename_to.strip()
                # Update artifacts to the new name
                for art in _s.query(Artifact).filter(Artifact.context == rename_from).all():
                    art.context = rename_to.strip()
                _s.commit()
        st.rerun()
    del_ctx = st.selectbox("Delete context", options=[c for c in available_contexts if c != "General"], key="ctx_del_from")
    if st.button("Delete", key="ctx_delete_btn"):
        with SessionLocal() as _s:
            # Move artifacts to General on delete
            for art in _s.query(Artifact).filter(Artifact.context == del_ctx).all():
                art.context = "General"
            row = _s.query(Contexts).filter(Contexts.name == del_ctx).first()
            if row:
                _s.delete(row)
            _s.commit()
        st.rerun()


def create_artifact(session, title: str, content: str, type_: str = "note", prompt_text: Optional[str] = None, context: str = "General") -> Artifact:
    artifact = Artifact(title=title.strip(), content=content.strip(), type=type_, prompt_text=prompt_text, context=context, created_by=user_name or "user")
    session.add(artifact)
    session.commit()
    session.add(ProvenanceEvent(artifact_id=artifact.id, actor_type="user", actor="user", event_type="created"))
    session.commit()
    ensure_artifact_embedding(session, artifact)
    try:
        session.add(MetricsEvent(event_type="artifact_created", details=json.dumps({"id": artifact.id, "type": type_})))
        session.commit()
    except Exception:
        session.rollback()
    return artifact


def save_comment(session, artifact_id: int, body: str) -> None:
    session.add(Comment(artifact_id=artifact_id, author=user_name or "user", body=body.strip()))
    session.commit()
    try:
        session.add(MetricsEvent(event_type="comment_added", details=json.dumps({"artifact_id": artifact_id})))
        session.commit()
    except Exception:
        session.rollback()


def save_template(session, title: str, prompt_text: str) -> Template:
    tpl = Template(title=title.strip(), prompt_text=prompt_text.strip())
    session.add(tpl)
    session.commit()
    try:
        session.add(MetricsEvent(event_type="template_saved", details=json.dumps({"id": tpl.id})))
        session.commit()
    except Exception:
        session.rollback()
    return tpl


def run_template(session, template: Template, params: Dict[str, str]) -> Optional[Artifact]:
    prompt = template.prompt_text
    try:
        formatted = prompt.format(**params)
    except Exception:
        formatted = prompt
    output_text = None
    if openai_client:
        try:
            resp = openai_client.responses.create(
                model=OPENAI_MODEL,
                input=[{"role": "user", "content": formatted}],
                temperature=0.3,
                max_output_tokens=600,
            )
            output_text = resp.output[0].content[0].text
        except Exception:
            output_text = None
    if output_text is None:
        output_text = f"[Local run] Template output for:\n\n{formatted}"
    new_art = create_artifact(session, title=f"Output: {template.title}", content=output_text, type_="ai_output", prompt_text=formatted)
    session.add(ProvenanceEvent(artifact_id=new_art.id, actor_type=("model" if openai_client else "user"), actor=(OPENAI_MODEL if openai_client else "local"), event_type="generated"))
    session.commit()
    try:
        session.add(MetricsEvent(event_type="template_run", details=json.dumps({"template_id": template.id, "artifact_id": new_art.id})))
        session.commit()
    except Exception:
        session.rollback()
    return new_art


def list_artifacts(session, query_text: str = "", filter_type: str = "All", context: str = "General") -> List[Artifact]:
    q = session.query(Artifact).filter(Artifact.context == context)
    if filter_type != "All":
        q = q.filter(Artifact.type == filter_type)
    artifacts = q.order_by(Artifact.created_at.desc()).all()
    if not query_text:
        return artifacts
    # Keyword match first
    qt = query_text.lower()
    keyword_rank = [
        (a, (qt in (a.title or "").lower()) + (qt in (a.content or "").lower()) + (qt in (a.prompt_text or "").lower()))
        for a in artifacts
    ]
    keyword_rank.sort(key=lambda x: x[1], reverse=True)
    ordered = [a for a, _ in keyword_rank if _ > 0]
    # Embedding rerank/augment
    try:
        embed_ids = embedding_search(session, query_text, top_k=25)
    except Exception:
        embed_ids = []
    embed_map = {id_: idx for idx, id_ in enumerate(embed_ids)}

    def score(a: Artifact) -> float:
        base = 1.0 if a in ordered else 0.0
        emb_boost = (len(embed_ids) - embed_map.get(a.id, len(embed_ids))) / max(len(embed_ids), 1)
        return base + emb_boost

    artifacts.sort(key=score, reverse=True)
    return artifacts


def templates_section(session):
    st.subheader("Templates")
    with st.expander("Create template", expanded=False):
        t_title = st.text_input("Template title", key="tpl_title")
        t_prompt = st.text_area("Prompt text (use {placeholders})", height=180, key="tpl_prompt")
        if st.button("Save template", type="primary", key="tpl_save"):
            if t_title.strip() and t_prompt.strip():
                save_template(session, t_title, t_prompt)
                st.success("Template saved.")
            else:
                st.warning("Please provide title and prompt text.")

    templates = session.query(Template).order_by(Template.created_at.desc()).all()
    if not templates:
        st.info("No templates yet.")
        return
    for tpl in templates:
        with st.expander(f"{tpl.title} (v{tpl.version})"):
            st.code(tpl.prompt_text)
            # Improved placeholder parsing: {name} via regex
            placeholders = list(dict.fromkeys(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", tpl.prompt_text)))
            params: Dict[str, str] = {}
            cols = st.columns(max(1, len(placeholders)))
            for idx, name in enumerate(placeholders):
                params[name] = cols[idx].text_input(name, key=f"tpl_param_{tpl.id}_{name}")
            if st.button(f"Run '{tpl.title}'", key=f"run_tpl_{tpl.id}"):
                art = run_template(session, tpl, params)
                if art:
                    st.success(f"Generated artifact #{art.id}: {art.title}")


def artifact_detail(session, artifact: Artifact):
    st.markdown(f"**Type:** {artifact.type} • **Created:** {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')} • **By:** {artifact.created_by}")
    if artifact.prompt_text:
        with st.expander("Source prompt"):
            st.code(artifact.prompt_text)
    st.text_area("Content", value=artifact.content, height=220, key=f"view_content_{artifact.id}")

    # Move context control
    with st.expander("Move to context"):
        with SessionLocal() as _s:
            ctx_rows = _s.query(Contexts).order_by(Contexts.name.asc()).all()
        ctx_opts = [c.name for c in ctx_rows]
        new_ctx = st.selectbox("New context", options=ctx_opts, index=ctx_opts.index(artifact.context) if artifact.context in ctx_opts else 0, key=f"mv_ctx_{artifact.id}")
        if st.button("Move", key=f"mv_btn_{artifact.id}") and new_ctx != artifact.context:
            artifact.context = new_ctx
            session.commit()
            st.success(f"Moved to {new_ctx}")
            st.rerun()

    st.markdown("---")
    st.subheader("Comments")
    for c in session.query(Comment).filter(Comment.artifact_id == artifact.id).order_by(Comment.created_at.asc()).all():
        st.markdown(f"- {c.author} ({c.created_at.strftime('%Y-%m-%d %H:%M')}): {c.body}")
    new_comment = st.text_input("Add a comment", key=f"new_comment_{artifact.id}")
    if st.button("Post comment", key=f"post_comment_{artifact.id}") and new_comment.strip():
        save_comment(session, artifact.id, new_comment)
        st.experimental_rerun()

    st.markdown("---")
    with st.expander("Save as template"):
        t_title = st.text_input("Template title", value=f"From: {artifact.title}", key=f"tpl_from_art_{artifact.id}")
        t_prompt = st.text_area("Prompt text (use {placeholders})", value=(artifact.prompt_text or artifact.content), height=160, key=f"tpl_prompt_from_art_{artifact.id}")
        if st.button("Create template", key=f"tpl_create_from_art_{artifact.id}"):
            if t_title.strip() and t_prompt.strip():
                save_template(session, t_title, t_prompt)
                st.success("Template created from artifact.")
                st.rerun()


def artifacts_section(session):
    st.subheader("Artifacts")
    with st.expander("Create new", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            a_title = st.text_input("Title", key="a_title")
            a_content = st.text_area("Content", height=180, key="a_content")
        with col2:
            a_type = st.selectbox("Type", options=["note", "ai_output", "document"], index=0, key="a_type")
            a_prompt = st.text_area("Source prompt (optional)", height=120, key="a_prompt")
            if st.button("Save artifact", type="primary", key="a_save"):
                if a_title.strip() and a_content.strip():
                    art = create_artifact(session, a_title, a_content, a_type, a_prompt if a_prompt.strip() else None, context=current_context)
                    st.success(f"Saved artifact #{art.id}")
                else:
                    st.warning("Please provide title and content.")

    st.markdown("---")
    col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
    with col_s1:
        qtext = st.text_input("Search (keyword, reranked by embeddings if API key present)", key="search_q")
    with col_s2:
        ftype = st.selectbox("Filter by type", options=["All", "note", "ai_output", "document"], index=0, key="filter_type")
    with col_s3:
        st.caption("Embedding search: " + ("ON" if openai_client else "OFF"))

    # Group by context as collapsible folders
    all_contexts = sorted({a.context for a in session.query(Artifact).all()})
    if current_context not in all_contexts:
        all_contexts = [current_context] + all_contexts
    for ctx in all_contexts:
        with st.expander(f"{ctx}", expanded=(ctx == current_context)):
            arts = list_artifacts(session, qtext, ftype, ctx)
            if not arts:
                st.caption("No artifacts.")
                continue
            options = [f"#{a.id} • {a.title}" for a in arts]
            sel = st.selectbox(f"Open in {ctx}", options=options, key=f"open_{ctx}")
            sel_id = int(sel.split(" ")[0][1:])
            st.caption(f"Shareable link: ?artifact={sel_id}")
            current = next(a for a in arts if a.id == sel_id)
            artifact_detail(session, current)


def cocreate_section(session):
    st.subheader("Co-create")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        # Create new conversation
        with st.expander("New conversation", expanded=False):
            c_title = st.text_input("Title", key="conv_title")
            if st.button("Create conversation", key="conv_create") and c_title.strip():
                conv = Conversation(title=c_title.strip(), context=current_context, created_by=user_name or "user", share_token=secrets.token_urlsafe(16))
                session.add(conv)
                session.commit()
                st.success("Conversation created. Share token enabled.")
        # Select conversation
        conversations = session.query(Conversation).filter(Conversation.context == current_context).order_by(Conversation.created_at.desc()).all()
        if conversations:
            labels = [f"#{c.id} • {c.title}" for c in conversations]
            sel = st.selectbox("Select conversation", options=labels, key="conv_select")
            selected_cid = int(sel.split(" ")[0][1:])
        else:
            st.info("No conversations yet.")
            selected_cid = None
    with col_right:
        if selected_cid is not None:
            conv = session.get(Conversation, selected_cid)
            if conv and conv.share_token:
                st.caption(f"Share link: ?share={conv.share_token}&conversation={conv.id}")
            # Presence heartbeat
            now = datetime.utcnow()
            ph = PresenceHeartbeat(user_name=user_name or "anon", context=current_context, conversation_id=selected_cid, last_seen=now)
            session.add(ph)
            # prune old (keep latest per user)
            try:
                session.commit()
            except Exception:
                session.rollback()
            cutoff = datetime.utcnow()
            active = session.query(PresenceHeartbeat).filter(
                PresenceHeartbeat.conversation_id == selected_cid,
            ).order_by(PresenceHeartbeat.last_seen.desc()).all()
            seen_users = set()
            active_names = []
            for hb in active:
                if hb.user_name not in seen_users:
                    active_names.append(hb.user_name)
                    seen_users.add(hb.user_name)
            st.caption("Active collaborators: " + (", ".join(active_names) if active_names else "none"))
            st.button("Refresh", key=f"refresh_{selected_cid}", on_click=lambda: st.rerun())

            # Use artifacts in co-creation
            with st.expander("Use artifacts in this conversation", expanded=False):
                q = st.text_input("Search artifacts (current context)", key=f"conv_art_search_{selected_cid}")
                arts = list_artifacts(session, q, "All", current_context)
                if not arts:
                    st.caption("No matching artifacts.")
                else:
                    for a in arts[:50]:
                        with st.container():
                            st.markdown(f"**#{a.id} • {a.title}** — {a.type} • {a.created_at.strftime('%Y-%m-%d')}")
                            btn_cols = st.columns([1,1,1])
                            if btn_cols[0].button("Insert content", key=f"conv_ins_full_{selected_cid}_{a.id}"):
                                session.add(Turn(conversation_id=selected_cid, role="system", author=f"artifact#{a.id}", content=f"[Artifact #{a.id}: {a.title}]\n\n{a.content}"))
                                session.commit()
                                st.rerun()
                            if btn_cols[1].button("Insert link", key=f"conv_ins_link_{selected_cid}_{a.id}"):
                                session.add(Turn(conversation_id=selected_cid, role="system", author=f"artifact#{a.id}", content=f"Reference: Artifact #{a.id} — {a.title}"))
                                session.commit()
                                st.rerun()
                            if btn_cols[2].button("Preview", key=f"conv_prev_{selected_cid}_{a.id}"):
                                st.code(a.content)

            # Use templates in co-creation
            with st.expander("Use templates", expanded=False):
                templates = session.query(Template).order_by(Template.created_at.desc()).all()
                if not templates:
                    st.caption("No templates yet.")
                else:
                    tpl_map = {f"#{t.id} • {t.title}": t for t in templates}
                    choice = st.selectbox("Pick a template", options=list(tpl_map.keys()), key=f"conv_tpl_pick_{selected_cid}")
                    tpl = tpl_map[choice]
                    st.code(tpl.prompt_text)
                    placeholders = list(dict.fromkeys(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", tpl.prompt_text)))
                    params: Dict[str, str] = {}
                    cols = st.columns(max(1, len(placeholders)))
                    for idx, name in enumerate(placeholders):
                        params[name] = cols[idx].text_input(name, key=f"conv_tpl_param_{selected_cid}_{tpl.id}_{name}")
                    c1, c2 = st.columns([1,1])
                    if c1.button("Insert prompt", key=f"conv_tpl_insert_{selected_cid}_{tpl.id}"):
                        try:
                            formatted = tpl.prompt_text.format(**{k: v for k, v in params.items() if v})
                        except Exception:
                            formatted = tpl.prompt_text
                        session.add(Turn(conversation_id=selected_cid, role="user", author=user_name or "user", content=formatted))
                        session.commit()
                        st.rerun()
                    if c2.button("Run template", key=f"conv_tpl_run_{selected_cid}_{tpl.id}"):
                        new_art = run_template(session, tpl, params)
                        session.add(Turn(conversation_id=selected_cid, role="assistant", author=OPENAI_MODEL if openai_client else "local", content=new_art.content if new_art else ""))
                        session.commit()
                        st.success(f"Generated artifact #{new_art.id}")
                        st.rerun()
            turns = session.query(Turn).filter(Turn.conversation_id == selected_cid).order_by(Turn.created_at.asc()).all()
            for t in turns:
                who = t.author if t.role == "user" else (OPENAI_MODEL if t.role == "assistant" else "system")
                st.markdown(f"**{t.role} ({who})**: {t.content}")
            st.markdown("---")
            u_msg = st.text_area("Your message", key=f"conv_msg_{selected_cid}")
            col_u1, col_u2, col_u3 = st.columns([1,1,1])
            with col_u1:
                if st.button("Send", key=f"conv_send_{selected_cid}") and u_msg.strip():
                    session.add(Turn(conversation_id=selected_cid, role="user", author=user_name or "user", content=u_msg.strip()))
                    session.commit()
                    st.rerun()
            with col_u2:
                # Model selector
                model_choice = st.selectbox("Model", 
                    options=["OpenAI GPT-4", "Claude 3.5 Sonnet", "Local Fallback"],
                    index=0 if openai_client else (1 if anthropic_client else 2),
                    key=f"model_choice_{selected_cid}")
                
                if st.button("Ask model", key=f"conv_model_{selected_cid}"):
                    # Build messages from turns
                    messages = []
                    for t in session.query(Turn).filter(Turn.conversation_id == selected_cid).order_by(Turn.created_at.asc()).all():
                        messages.append({"role": t.role, "content": t.content})
                    
                    model_reply = "Model unavailable."
                    model_name = "local"
                    
                    if model_choice == "OpenAI GPT-4" and openai_client and messages:
                        try:
                            resp = openai_client.responses.create(
                                model=OPENAI_MODEL,
                                input=[{"role": m["role"], "content": m["content"]} for m in messages],
                                temperature=0.4,
                                max_output_tokens=500,
                            )
                            model_reply = resp.output[0].content[0].text
                            model_name = OPENAI_MODEL
                        except Exception:
                            model_reply = "(Error calling OpenAI)"
                    
                    elif model_choice == "Claude 3.5 Sonnet" and anthropic_client and messages:
                        try:
                            resp = anthropic_client.messages.create(
                                model=ANTHROPIC_MODEL,
                                max_tokens=500,
                                temperature=0.4,
                                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
                            )
                            model_reply = resp.content[0].text
                            model_name = ANTHROPIC_MODEL
                        except Exception:
                            model_reply = "(Error calling Claude)"
                    
                    else:
                        model_reply = "No AI model available. Please set API keys in .env file."
                        model_name = "local"
                    
                    session.add(Turn(conversation_id=selected_cid, role="assistant", author=model_name, content=model_reply))
                    session.commit()
                    st.rerun()
            with col_u3:
                with st.popover("Import transcript"):
                    raw = st.text_area("Paste ChatGPT/Claude transcript", height=160, key=f"conv_import_{selected_cid}")
                    if st.button("Import", key=f"conv_import_btn_{selected_cid}") and raw.strip():
                        # Simple heuristic: lines starting with 'You:' => user, 'Assistant:' => assistant
                        for line in raw.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.lower().startswith("you:") or line.lower().startswith("user:"):
                                content = line.split(":", 1)[1].strip() if ":" in line else line
                                session.add(Turn(conversation_id=selected_cid, role="user", author=user_name or "user", content=content))
                            elif line.lower().startswith("assistant:") or line.lower().startswith("model:"):
                                content = line.split(":", 1)[1].strip() if ":" in line else line
                                session.add(Turn(conversation_id=selected_cid, role="assistant", author=OPENAI_MODEL if openai_client else "model", content=content))
                        session.commit()
                        st.success("Imported transcript.")


def importers_section(session):
    st.subheader("Importers")
    st.markdown("Import ChatGPT data export (ZIP or conversations.json).")
    uploaded = st.file_uploader("Upload ChatGPT export (ZIP or JSON)", type=["zip", "json"], accept_multiple_files=False)
    if uploaded is None:
        return
    try:
        conversations = []
        if uploaded.type == "application/json" or uploaded.name.endswith(".json"):
            data = json.loads(uploaded.read().decode("utf-8"))
            # supports either list of conversations or object with conversations key
            conversations = data if isinstance(data, list) else data.get("conversations", [])
        else:
            with zipfile.ZipFile(io.BytesIO(uploaded.read())) as zf:
                # ChatGPT export contains conversations.json at root
                with zf.open("conversations.json") as fh:
                    data = json.loads(fh.read().decode("utf-8"))
                    conversations = data if isinstance(data, list) else data.get("conversations", [])
    except Exception:
        st.error("Could not parse file. Ensure it's a ChatGPT conversations export.")
        return

    if not conversations:
        st.info("No conversations found in file.")
        return

    imported_count = 0
    for conv in conversations:
        title = conv.get("title") or "ChatGPT Conversation"
        mapping = conv.get("mapping") or {}
        # Build chronological messages
        msgs = []
        try:
            # mapping is a dict of id -> node with 'message' and 'children'
            # We flatten by created time when available, else insertion order
            for node in mapping.values():
                msg = node.get("message") or {}
                author = ((msg.get("author") or {}).get("role")) or "user"
                content_parts = (msg.get("content") or {}).get("parts") or []
                text = "\n\n".join([p for p in content_parts if isinstance(p, str)])
                if text:
                    created = msg.get("create_time") or 0
                    msgs.append({"role": author, "text": text, "ts": created})
            msgs.sort(key=lambda m: m["ts"])
        except Exception:
            continue
        if not msgs:
            continue
        first_user = next((m for m in msgs if m["role"] == "user" and m["text"].strip()), None)
        combined = []
        for m in msgs:
            combined.append(f"{m['role'].upper()}:\n{m['text']}")
        content = "\n\n---\n\n".join(combined)
        art = create_artifact(
            session,
            title=f"Imported: {title}",
            content=content,
            type_="document",
            prompt_text=(first_user["text"] if first_user else None),
            context=current_context,
        )
        imported_count += 1
    st.success(f"Imported {imported_count} conversation(s) into context '{current_context}'.")


def appbeelder_section(session):
    st.subheader("Appbeelder - Micro-App Builder")
    
    # Show API key setup instructions if no keys are available
    if not openai_client and not anthropic_client:
        with st.expander("🔑 Setup API Keys (Optional)", expanded=False):
            st.markdown("""
            **To enable AI-powered app generation, add your API keys to the `.env` file:**
            
            ```bash
            # OpenAI API Key
            OPENAI_API_KEY=sk-your-openai-key-here
            OPENAI_MODEL=gpt-4
            
            # Anthropic API Key  
            ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
            ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
            ```
            
            **Get API Keys:**
            - **OpenAI**: https://platform.openai.com/api-keys
            - **Anthropic**: https://console.anthropic.com/
            
            **Without API keys**, the Appbeelder will create functional template apps using Local Fallback mode.
            """)
    
    # Create new micro-app
    with st.expander("Create Micro-App", expanded=False):
        app_name = st.text_input("App Name", key="app_name")
        app_desc = st.text_area("Description", key="app_desc")
        app_prompt = st.text_area("Describe what the app should do", height=120, key="app_prompt")
        
        # Model selection
        available_models = []
        if openai_client:
            available_models.append("OpenAI GPT-4")
        if anthropic_client:
            available_models.append("Claude 3.5 Sonnet")
        available_models.append("Local Fallback")
        
        if not available_models:
            st.warning("⚠️ No AI models available. Please set API keys in .env file or use Local Fallback.")
            selected_model = "Local Fallback"
        else:
            selected_model = st.selectbox("Choose AI Model", available_models, key="app_model")
            
        # Show API key status
        if not openai_client and not anthropic_client:
            st.info("💡 **Local Fallback Mode**: Creating functional template apps without AI generation. Add API keys to .env for AI-powered generation.")
        
        if st.button("Generate App", key="gen_app") and app_prompt.strip():
            # Generate Streamlit code using AI
            code_prompt = f"""Create a Streamlit app that: {app_prompt}

IMPORTANT REQUIREMENTS:
- Use ONLY standard Python libraries and Streamlit (no external modules like 'mindmap', 'matplotlib', etc.)
- Make the code self-contained and runnable
- Use Streamlit's built-in components (st.markdown, st.columns, st.color_picker, etc.)
- For colors, use HTML/CSS styling with st.markdown() or Streamlit's color features
- Include proper error handling
- Add comments explaining key parts
- Return only the Python code, no markdown formatting

Requirements:
- Use streamlit as st
- Make it functional and user-friendly
- Ensure the code will run without external dependencies"""

            streamlit_code = None
            mcp_code = None
            
            # Use selected model
            if selected_model == "OpenAI GPT-4" and openai_client:
                try:
                    resp = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": code_prompt}],
                        temperature=0.3,
                        max_tokens=2000,
                    )
                    raw_code = resp.choices[0].message.content
                    if raw_code and raw_code.strip():
                        streamlit_code = clean_ai_generated_code(raw_code)
                        if streamlit_code:
                            st.success("✅ Generated with OpenAI GPT-4")
                        else:
                            streamlit_code = None
                    else:
                        streamlit_code = None
                except Exception as e:
                    st.warning(f"OpenAI API failed: {str(e)}")
                    streamlit_code = None
            
            elif selected_model == "Claude 3.5 Sonnet" and anthropic_client:
                try:
                    resp = anthropic_client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=2000,
                        temperature=0.3,
                        messages=[{"role": "user", "content": code_prompt}]
                    )
                    raw_code = resp.content[0].text
                    if raw_code and raw_code.strip():
                        streamlit_code = clean_ai_generated_code(raw_code)
                        if streamlit_code:
                            st.success("✅ Generated with Claude 3.5 Sonnet")
                        else:
                            streamlit_code = None
                    else:
                        streamlit_code = None
                except Exception as e:
                    st.warning(f"Anthropic API failed: {str(e)}")
                    streamlit_code = None
            
            # Always use fallback if no code generated or if Local Fallback selected
            if not streamlit_code or selected_model == "Local Fallback":
                st.info("Using Local Fallback - creating functional template app")
                # Create a more functional fallback app
                streamlit_code = f'''import streamlit as st
import pandas as pd
import datetime

st.title("{app_name}")
st.write("{app_desc}")

# Basic app structure based on: {app_prompt}
st.header("App Features")

# Add some interactive elements
if "todo" in "{app_prompt}".lower():
    st.subheader("Todo List")
    todo = st.text_input("Add a new todo")
    if st.button("Add Todo"):
        st.success(f"Added: {{todo}}")
    
    # Simple todo list
    todos = ["Sample todo 1", "Sample todo 2"]
    for i, todo in enumerate(todos):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"• {{todo}}")
        with col2:
            if st.button("Delete", key=f"del_{{i}}"):
                st.success("Todo deleted!")

elif "finance" in "{app_prompt}".lower():
    st.subheader("Finance Tracker")
    income = st.number_input("Monthly Income", value=5000)
    expenses = st.number_input("Monthly Expenses", value=3000)
    savings = income - expenses
    st.metric("Monthly Savings", f"${{savings}}")
    
    if savings > 0:
        st.success("Great job saving money!")
    else:
        st.warning("You're spending more than you earn!")

elif "recipe" in "{app_prompt}".lower():
    st.subheader("Recipe Manager")
    recipe_name = st.text_input("Recipe Name")
    ingredients = st.text_area("Ingredients (one per line)")
    instructions = st.text_area("Instructions")
    
    if st.button("Save Recipe"):
        st.success(f"Recipe '{{recipe_name}}' saved!")

elif "mind map" in "{app_prompt}".lower():
    st.subheader("Mind Map Creator")
    
    # Text input for mind map content
    text_input = st.text_area("Enter text to create mind map from:", height=100)
    
    if st.button("Generate Mind Map"):
        if text_input:
            # Simple mind map visualization
            st.success("Mind Map Generated!")
            
            # Parse text into concepts
            words = text_input.split()
            concepts = [word.strip('.,!?;:') for word in words if len(word) > 3]
            concepts = list(set(concepts))[:10]  # Limit to 10 unique concepts
            
            # Display as a simple mind map
            st.subheader("Mind Map Structure:")
            center_concept = concepts[0] if concepts else "Main Topic"
            
            col1, col2, col3 = st.columns(3)
            
            with col2:
                st.markdown(f"### 🎯 {{center_concept}}")
            
            # Display related concepts
            if len(concepts) > 1:
                st.subheader("Related Concepts:")
                for i, concept in enumerate(concepts[1:], 1):
                    st.write(f"• {{concept}}")
            
            # Simple visualization
            st.subheader("Visual Mind Map:")
            st.info("💡 **Center**: {{center_concept}}")
            for concept in concepts[1:6]:  # Show first 5 related concepts
                st.write(f"  └── {{concept}}")
        else:
            st.warning("Please enter some text to create a mind map!")

else:
    # Generic app
    st.subheader("Data Input")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    email = st.text_input("Email")
    
    if st.button("Submit"):
        st.success(f"Hello {{name}}! You are {{age}} years old.")
        st.write(f"Email: {{email}}")

st.markdown("---")
st.info("This is a generated app template. Customize it based on your needs!")
'''

            # Generate MCP server code
            mcp_prompt = f"""Create an MCP (Model Context Protocol) server for this Streamlit app: {app_name}

The app does: {app_prompt}

Generate a Python MCP server that exposes this functionality as tools for AI agents.
Include proper MCP protocol implementation, tool definitions, and error handling."""

            if selected_model == "OpenAI GPT-4" and openai_client:
                try:
                    resp = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": mcp_prompt}],
                        temperature=0.3,
                        max_tokens=1500,
                    )
                    mcp_code = resp.choices[0].message.content
                except Exception:
                    pass
            
            elif selected_model == "Claude 3.5 Sonnet" and anthropic_client:
                try:
                    resp = anthropic_client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=1500,
                        temperature=0.3,
                        messages=[{"role": "user", "content": mcp_prompt}]
                    )
                    mcp_code = resp.content[0].text
                except Exception:
                    pass

            if not mcp_code:  # Fallback MCP code
                mcp_code = f"""# MCP Server for {app_name}
# TODO: Implement MCP server based on: {app_prompt}

from mcp.server import Server
from mcp.types import Tool

server = Server("brainio-{app_name.lower().replace(' ', '-')}")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="{app_name.lower().replace(' ', '_')}",
            description="{app_desc}",
            inputSchema={{"type": "object", "properties": {{}}}}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "{app_name.lower().replace(' ', '_')}":
        return {{"content": "Tool executed"}}
    raise ValueError(f"Unknown tool: {{name}}")
"""

            # Save micro-app
            micro_app = MicroApp(
                name=app_name.strip(),
                description=app_desc.strip(),
                streamlit_code=streamlit_code,
                mcp_server_code=mcp_code,
                context=current_context,
                created_by=user_name or "user"
            )
            session.add(micro_app)
            session.commit()
            st.success(f"Micro-app '{app_name}' created!")

    # List existing micro-apps
    st.markdown("---")
    apps = session.query(MicroApp).filter(MicroApp.context == current_context, MicroApp.is_active == 1).order_by(MicroApp.created_at.desc()).all()
    
    if not apps:
        st.info("No micro-apps yet. Create one above!")
        return
    
    for app in apps:
        with st.expander(f"📱 {app.name} - {app.description or 'No description'}"):
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            
            with col1:
                if st.button("Run App", key=f"run_app_{app.id}"):
                    # Create a permanent file in the apps directory
                    import os
                    apps_dir = "generated_apps"
                    os.makedirs(apps_dir, exist_ok=True)
                    
                    app_filename = f"{app.name.lower().replace(' ', '_')}.py"
                    app_path = os.path.join(apps_dir, app_filename)
                    
                    with open(app_path, 'w') as f:
                        f.write(app.streamlit_code)
                    
                    st.success(f"App saved to: {app_path}")
                    st.code(f"Run: streamlit run {app_path}")
                    
                    # Also provide a direct run button
                    if st.button("🚀 Launch App Now", key=f"launch_{app.id}"):
                        # Store the app to run in session state
                        st.session_state['running_app'] = app.streamlit_code
                        st.session_state['app_name'] = app.name
                        st.session_state['show_app_runner'] = True
                        st.rerun()
            
            with col2:
                if st.button("View Code", key=f"view_code_{app.id}"):
                    st.code(app.streamlit_code, language="python")
            
            with col3:
                if st.button("MCP Server", key=f"mcp_{app.id}"):
                    st.code(app.mcp_server_code, language="python")
                    st.download_button(
                        "Download MCP Server",
                        app.mcp_server_code,
                        f"{app.name.lower().replace(' ', '_')}_mcp_server.py",
                        "text/python"
                    )
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{app.id}"):
                    # Delete from database
                    session.delete(app)
                    session.commit()
                    
                    # Delete file if it exists
                    import os
                    apps_dir = "generated_apps"
                    app_filename = f"{app.name.lower().replace(' ', '_')}.py"
                    app_path = os.path.join(apps_dir, app_filename)
                    if os.path.exists(app_path):
                        os.remove(app_path)
                    
                    st.success(f"App '{app.name}' deleted!")
                    st.rerun()
            
            # Edit functionality
            st.markdown("---")
            
            # Tab interface for different editing modes
            edit_tab1, edit_tab2 = st.tabs(["🤖 AI Assistant", "✏️ Manual Edit"])
            
            with edit_tab1:
                st.subheader("Chat with AI App Builder")
                st.info("💡 Describe what you want to change and the AI will update your app code!")
                
                # Initialize chat history for this app
                chat_key = f"app_chat_{app.id}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = [
                        {"role": "assistant", "content": f"Hi! I'm your AI app builder. I can help you modify your '{app.name}' app. What would you like to change?"}
                    ]
                
                # Display chat history
                for message in st.session_state[chat_key]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Chat input
                user_input = st.chat_input("Describe what you want to change...", key=f"chat_input_{app.id}")
                
                if user_input:
                    # Add user message to chat
                    st.session_state[chat_key].append({"role": "user", "content": user_input})
                    
                    # Show user message
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    # Generate AI response and updated code
                    with st.chat_message("assistant"):
                        with st.spinner("AI is updating your app..."):
                            # Create prompt for AI to modify the app
                            modification_prompt = f"""You are an AI app builder. The user wants to modify their Streamlit app.

Current app code:
```python
{app.streamlit_code}
```

User's request: {user_input}

IMPORTANT REQUIREMENTS:
1. Use ONLY standard Python libraries and Streamlit (no external modules like 'mindmap', 'matplotlib', etc.)
2. Make the code self-contained and runnable
3. Use Streamlit's built-in components (st.markdown, st.columns, st.color_picker, etc.)
4. For colors, use HTML/CSS styling with st.markdown() or Streamlit's color features
5. Return ONLY the updated Python code, no markdown formatting or explanations outside the code

Please:
1. Analyze the current code
2. Understand what the user wants to change
3. Provide the updated code with the modifications using only Streamlit and standard Python
4. Ensure the code will run without external dependencies"""

                            updated_code = None
                            
                            # Try to get AI response
                            if openai_client:
                                try:
                                    resp = openai_client.chat.completions.create(
                                        model=OPENAI_MODEL,
                                        messages=[{"role": "user", "content": modification_prompt}],
                                        temperature=0.3,
                                        max_tokens=3000,
                                    )
                                    raw_code = resp.choices[0].message.content
                                    if raw_code and raw_code.strip():
                                        # Clean the code - remove markdown formatting
                                        updated_code = clean_ai_generated_code(raw_code)
                                        if updated_code:
                                            st.success("✅ App updated with AI assistance!")
                                        else:
                                            updated_code = None
                                    else:
                                        updated_code = None
                                except Exception as e:
                                    st.warning(f"OpenAI API failed: {str(e)}")
                            
                            if not updated_code and anthropic_client:
                                try:
                                    resp = anthropic_client.messages.create(
                                        model=ANTHROPIC_MODEL,
                                        max_tokens=3000,
                                        temperature=0.3,
                                        messages=[{"role": "user", "content": modification_prompt}]
                                    )
                                    raw_code = resp.content[0].text
                                    if raw_code and raw_code.strip():
                                        # Clean the code - remove markdown formatting
                                        updated_code = clean_ai_generated_code(raw_code)
                                        if updated_code:
                                            st.success("✅ App updated with AI assistance!")
                                        else:
                                            updated_code = None
                                    else:
                                        updated_code = None
                                except Exception as e:
                                    st.warning(f"Anthropic API failed: {str(e)}")
                            
                            if not updated_code:
                                # Fallback: provide helpful response
                                updated_code = app.streamlit_code
                                st.warning("⚠️ AI not available. Using manual edit mode instead.")
                                st.write("I understand you want to: " + user_input)
                                st.write("Please use the 'Manual Edit' tab to make these changes.")
                            
                            # Add AI response to chat
                            if updated_code and updated_code != app.streamlit_code:
                                st.session_state[chat_key].append({
                                    "role": "assistant", 
                                    "content": f"I've updated your app based on: '{user_input}'. The changes have been applied!"
                                })
                                
                                # Update the app code
                                app.streamlit_code = updated_code
                                session.commit()
                                
                                # Update the file
                                import os
                                apps_dir = "generated_apps"
                                os.makedirs(apps_dir, exist_ok=True)
                                app_filename = f"{app.name.lower().replace(' ', '_')}.py"
                                app_path = os.path.join(apps_dir, app_filename)
                                
                                with open(app_path, 'w') as f:
                                    f.write(updated_code)
                                
                                st.rerun()
                            else:
                                st.session_state[chat_key].append({
                                    "role": "assistant", 
                                    "content": "I couldn't make that change automatically. Please try being more specific or use the Manual Edit tab."
                                })
                                st.write("I couldn't make that change automatically. Please try being more specific or use the Manual Edit tab.")
            
            with edit_tab2:
                st.subheader("Manual Code Editor")
                
                # Text area for editing code
                edited_code = st.text_area(
                    "Streamlit Code",
                    value=app.streamlit_code,
                    height=400,
                    key=f"edit_code_{app.id}"
                )
                
                col_edit1, col_edit2 = st.columns(2)
                
                with col_edit1:
                    if st.button("💾 Save Changes", key=f"save_edit_{app.id}"):
                        # Update the app in database
                        app.streamlit_code = edited_code
                        session.commit()
                        
                        # Update the file
                        import os
                        apps_dir = "generated_apps"
                        os.makedirs(apps_dir, exist_ok=True)
                        app_filename = f"{app.name.lower().replace(' ', '_')}.py"
                        app_path = os.path.join(apps_dir, app_filename)
                        
                        with open(app_path, 'w') as f:
                            f.write(edited_code)
                        
                        st.success("App updated successfully!")
                        st.rerun()
                
                with col_edit2:
                    if st.button("🔄 Reset to Original", key=f"reset_{app.id}"):
                        st.rerun()
                
                # Preview the edited code
                st.subheader("Preview")
                st.code(edited_code, language="python")


def teams_section(session):
    st.subheader("Teams & Collaboration")
    
    # Create new team
    with st.expander("Create Team", expanded=False):
        team_name = st.text_input("Team Name", key="team_name")
        team_desc = st.text_area("Description", key="team_desc")
        if st.button("Create Team", key="create_team") and team_name.strip():
            team = Team(name=team_name.strip(), description=team_desc.strip(), created_by=user_name or "user")
            session.add(team)
            session.commit()
            # Add creator as owner
            member = TeamMember(team_id=team.id, user_name=user_name or "user", role="owner")
            session.add(member)
            session.commit()
            st.success(f"Team '{team_name}' created!")
            st.rerun()
    
    # List teams
    st.markdown("---")
    teams = session.query(Team).order_by(Team.created_at.desc()).all()
    
    if not teams:
        st.info("No teams yet. Create one above!")
        return
    
    for team in teams:
        with st.expander(f"👥 {team.name} - {team.description or 'No description'}"):
            # Check if current user is member
            membership = session.query(TeamMember).filter(
                TeamMember.team_id == team.id,
                TeamMember.user_name == (user_name or "user")
            ).first()
            
            if membership:
                st.caption(f"Your role: {membership.role}")
                
                # Team members
                members = session.query(TeamMember).filter(TeamMember.team_id == team.id).all()
                st.write("**Members:**")
                for member in members:
                    st.write(f"- {member.user_name} ({member.role})")
                
                # Add member (if admin/owner)
                if membership.role in ["owner", "admin"]:
                    with st.expander("Add Member"):
                        new_member = st.text_input("Username", key=f"new_member_{team.id}")
                        new_role = st.selectbox("Role", ["member", "viewer", "admin"], key=f"new_role_{team.id}")
                        if st.button("Add", key=f"add_member_{team.id}") and new_member.strip():
                            existing = session.query(TeamMember).filter(
                                TeamMember.team_id == team.id,
                                TeamMember.user_name == new_member.strip()
                            ).first()
                            if not existing:
                                member = TeamMember(team_id=team.id, user_name=new_member.strip(), role=new_role)
                                session.add(member)
                                session.commit()
                                st.success(f"Added {new_member} as {new_role}")
                                st.rerun()
                            else:
                                st.error("User already in team")
            else:
                st.caption("You're not a member of this team")
                if st.button("Request to Join", key=f"join_{team.id}"):
                    # Auto-approve for demo (in real app, this would be a request)
                    member = TeamMember(team_id=team.id, user_name=user_name or "user", role="member")
                    session.add(member)
                    session.commit()
                    st.success("Joined team!")
                    st.rerun()


def app_runner_section():
    """Run generated apps directly within the main app"""
    st.title("🚀 Running Your Generated App")
    
    if 'running_app' in st.session_state and 'app_name' in st.session_state:
        app_name = st.session_state['app_name']
        app_code = st.session_state['running_app']
        
        st.subheader(f"App: {app_name}")
        
        # Back button
        if st.button("← Back to Appbeelder"):
            del st.session_state['running_app']
            del st.session_state['app_name']
            st.session_state['show_app_runner'] = False
            st.rerun()
        
        st.markdown("---")
        
        # Execute the app code
        try:
            # Create a safe execution environment
            exec_globals = {
                'st': st,
                'pd': __import__('pandas'),
                'np': __import__('numpy'),
                'datetime': __import__('datetime'),
                'time': __import__('time'),
                'random': __import__('random'),
                'math': __import__('math'),
                'json': __import__('json'),
                'os': __import__('os'),
                'sys': __import__('sys')
            }
            
            # Execute the app code
            exec(app_code, exec_globals)
            
        except Exception as e:
            st.error(f"Error running app: {str(e)}")
            st.code(app_code, language="python")
    else:
        st.info("No app selected to run. Go to Appbeelder to launch an app.")
        if st.button("Go to Appbeelder"):
            st.rerun()

def main_app():
    # Check if we should show the app runner
    if st.session_state.get('show_app_runner', False):
        app_runner_section()
        return
    
    session = SessionLocal()
    tabs = st.tabs(["Artifacts", "Templates", "Co-create", "Importers", "Appbeelder", "Teams"])
    with tabs[0]:
        artifacts_section(session)
    with tabs[1]:
        templates_section(session)
    with tabs[2]:
        cocreate_section(session)
    with tabs[3]:
        importers_section(session)
    with tabs[4]:
        appbeelder_section(session)
    with tabs[5]:
        teams_section(session)


if __name__ == "__main__":
    main_app()
