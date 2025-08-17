import io
import json
import mimetypes
import uuid
from typing import List, Tuple

import requests
import streamlit as st

st.set_page_config(
    page_title="Multi-Agents Chat",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_API_BASE = "http://localhost:8000"
api_base = DEFAULT_API_BASE 

with st.sidebar:
    st.title("📎 Documents & Index")
    #api_base = st.text_input("API base URL", value=DEFAULT_API_BASE, help="Your FastAPI server URL")
    #st.caption("Endpoints used: `/query`, `/embeddings/upload`, `/embeddings/reset`")

# -------------------- Session state --------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:12]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_final" not in st.session_state:
    st.session_state.last_final = None

SUPPORTED_EXTS = ["pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "docx", "doc"]

def _build_multipart(files: List[io.BytesIO], names: List[str]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    """Build the `files` list for requests.post(..., files=...)"""
    payload = []
    for fobj, fname in zip(files, names):
        fbytes = fobj.getvalue()
        mime = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        payload.append(("files", (fname, fbytes, mime)))
    return payload

def _post_query(base: str, thread_id: str, query: str) -> dict:
    url = f"{base.rstrip('/')}/query"
    resp = requests.post(url, json={"thread_id": thread_id, "query": query}, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"/query failed: {resp.status_code} {resp.text}")
    return resp.json()

def _post_upload(base: str, files_payload) -> dict:
    url = f"{base.rstrip('/')}/embeddings/upload"
    resp = requests.post(url, files=files_payload, timeout=600)
    if not resp.ok:
        raise RuntimeError(f"/embeddings/upload failed: {resp.status_code} {resp.text}")
    return resp.json()

def _delete_reset(base: str) -> dict:
    url = f"{base.rstrip('/')}/embeddings/reset"
    resp = requests.delete(url, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"/embeddings/reset failed: {resp.status_code} {resp.text}")
    return resp.json()

def _pick_assistant_reply(final_payload: dict) -> str:
    """
    Prefer Agent2 (RAG) -> Agent1 (summary) -> Agent3 (internet).
    Fall back to a compact JSON if nothing obvious is present.
    """
    agent = final_payload.get("agent_responses", {})
    a2 = agent.get("Agent2", {})
    a1 = agent.get("Agent1", {})
    a3 = agent.get("Agent3", {})

    # Agent2
    try:
        d2 = a2.get("data") or {}
        if d2.get("response"):
            return d2["response"]
    except Exception:
        pass

    # Agent1
    try:
        d1 = a1.get("data") or {}
        if d1.get("document_summary"):
            return d1["document_summary"]
    except Exception:
        pass

    # Agent3
    try:
        d3 = a3.get("data") or {}
        if d3.get("response"):
            return d3["response"]
    except Exception:
        pass

    return json.dumps(final_payload.get("manager_agent", {}), ensure_ascii=False)

def _render_sources(final_payload: dict):
    agent = final_payload.get("agent_responses", {})
    a3 = agent.get("Agent3", {})
    try:
        data = a3.get("data") or {}
        sources = data.get("source") or []
        if sources:
            with st.expander("Sources"):
                for u in sources:
                    st.write(f"- {u}")
    except Exception:
        pass

with st.sidebar:
    st.subheader("Upload to Build Embeddings")
    uploaded = st.file_uploader(
        "Choose files",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True,
        help="PDF, TXT, MD, CSV/TSV, XLSX/XLS, DOCX/DOC",
    )

    col_u1, col_u2 = st.columns([1, 1])
    with col_u1:
        do_upload = st.button("Upload & Build", use_container_width=True, type="primary")
    with col_u2:
        new_chat = st.button("New Chat", use_container_width=True, help="Start a fresh thread while keeping the index")

    st.divider()
    st.subheader("Reset")
    do_reset_all = st.button("Clear Previous Data", use_container_width=True)


if do_upload:
    if not uploaded:
        st.sidebar.error("Please select at least one file.")
    else:
        try:
            files_payload = _build_multipart(
                files=[io.BytesIO(f.getvalue()) for f in uploaded],
                names=[f.name for f in uploaded],
            )
            with st.spinner("Uploading and building embeddings..."):
                result = _post_upload(api_base, files_payload)
            st.sidebar.success("Embeddings built successfully.")
            with st.sidebar.expander("Upload result", expanded=False):
                st.json(result)
        except Exception as e:
            st.sidebar.error(str(e))

if do_reset_all:
    try:
        with st.spinner("Clearing previous data..."):
            result = _delete_reset(api_base)
        st.sidebar.success("Previous data cleared.")
        with st.sidebar.expander("Reset result", expanded=False):
            st.json(result)
    except Exception as e:
        st.sidebar.error(str(e))

if new_chat:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    st.session_state.messages = []
    st.session_state.last_final = None
    st.sidebar.success("Started a new thread.")


st.title("🧭 Multi-Agents Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask about your documents (or general queries)…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                payload = _post_query(api_base, st.session_state.thread_id, prompt)
                final = payload
                st.session_state.last_final = final
                reply = _pick_assistant_reply(final)
                st.markdown(reply)
                _render_sources(final)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        err = f"❌ {e}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})

st.caption(
    f"Thread: `{st.session_state.thread_id}` · Talking to `{api_base}` · "
    "Use the sidebar to upload files (build embeddings) or clear previous data."
)
