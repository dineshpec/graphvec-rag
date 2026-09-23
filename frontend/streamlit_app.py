"""Production Streamlit UI for the Hybrid RAG assistant.

Design goals:
- Clean, uncluttered chat experience by default (no raw guardrail JSON
  shoved into every message).
- Guardrail outcome shown as a small, unobtrusive status badge; full
  technical detail (reason, retrieval metadata) is opt-in via a
  "Show technical details" toggle, useful for admins/debugging.
- Core actions (upload, manage documents, connection settings) organized
  into tabs instead of a long, noisy sidebar.
"""
import os
import requests
import streamlit as st

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_API_KEY_DEFAULT = os.getenv("BACKEND_API_KEY", "")

try:
    UPLOAD_TIMEOUT = float(os.getenv("UPLOAD_TIMEOUT_SECONDS", "600"))
    if UPLOAD_TIMEOUT <= 0:
        raise ValueError
except ValueError:
    raise RuntimeError("UPLOAD_TIMEOUT_SECONDS must be a positive number.")
QUERY_TIMEOUT = 60
HEALTH_TIMEOUT = 5
DOCS_TIMEOUT = 15

# Guardrail status -> (badge icon, short label, color) for the compact view.
GUARDRAIL_BADGES = {
    "PASSED": ("✅", "Verified", "green"),
    "Blocked by Input Guardrail": ("🚫", "Blocked", "red"),
    "Out of Scope": ("❔", "Out of scope", "gray"),
    "Flagged by Output Guardrail": ("⚠️", "Needs review", "orange"),
}

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------
# Session state
# ----------------------------------------------------
st.session_state.setdefault("api_key", BACKEND_API_KEY_DEFAULT)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("documents", None)
st.session_state.setdefault("show_debug", False)


def _headers() -> dict:
    """Builds request headers, attaching the API key if one is configured."""
    if st.session_state.get("api_key"):
        return {"X-API-Key": st.session_state.api_key}
    return {}


def _extract_error_detail(response: requests.Response) -> str:
    """Parses a backend error response's JSON 'detail' field, falling back to raw text."""
    try:
        data = response.json()
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
        return response.text
    except ValueError:
        return response.text


def fetch_documents():
    """Fetches the list of ingested documents from the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/v1/documents", headers=_headers(), timeout=DOCS_TIMEOUT)
        if resp.status_code == 200:
            st.session_state.documents = resp.json()
            return True, None
        return False, f"Failed to load documents ({resp.status_code}): {_extract_error_detail(resp)}"
    except requests.exceptions.Timeout:
        return False, "Timed out while loading documents."
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the backend to load documents."
    except requests.exceptions.RequestException as e:
        return False, f"Error loading documents: {str(e)}"


def check_backend_status():
    """Checks /health and /ready endpoints without crashing if the backend is unreachable."""
    health_ok = False
    ready_info = None
    error_msg = None
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=HEALTH_TIMEOUT)
        health_ok = health_resp.status_code == 200
    except requests.exceptions.Timeout:
        error_msg = "Health check timed out."
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to the backend."
    except requests.exceptions.RequestException as e:
        error_msg = f"Health check error: {str(e)}"

    if health_ok:
        try:
            ready_resp = requests.get(f"{BACKEND_URL}/ready", timeout=HEALTH_TIMEOUT)
            ready_info = ready_resp.json()
        except requests.exceptions.RequestException:
            ready_info = None

    return health_ok, ready_info, error_msg


def render_guardrail_badge(metadata: dict) -> None:
    """Renders a single-line, unobtrusive guardrail status badge."""
    if not metadata:
        return
    status = metadata.get("status", "UNKNOWN")
    icon, label, _color = GUARDRAIL_BADGES.get(status, ("ℹ️", status, "gray"))
    st.caption(f"{icon} {label}")


def render_guardrail_details(metadata: dict) -> None:
    """Renders full guardrail/debug detail, only called when debug mode is on."""
    if not metadata:
        return
    with st.expander("Technical details", expanded=False):
        st.write(f"**Guardrail status:** `{metadata.get('status')}`")
        if metadata.get("reason"):
            st.write(f"**Reason:** {metadata.get('reason')}")
        if metadata.get("retrieval_metadata"):
            st.json(metadata["retrieval_metadata"])


# ----------------------------------------------------
# Sidebar: connection + preferences only (kept minimal)
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Hybrid RAG Assistant")
    st.caption("FAISS + Neo4j AuraDB + OpenAI")

    st.session_state.api_key = st.text_input(
        "Backend API Key",
        value=st.session_state.api_key,
        type="password",
        help="Sent as the X-API-Key header. Leave blank if the backend has auth disabled.",
    )

    if st.button("Check Backend Status", use_container_width=True):
        health_ok, ready_info, error_msg = check_backend_status()
        if error_msg:
            st.error(f"Unreachable: {error_msg}")
        elif not health_ok:
            st.error("Health check failed.")
        elif ready_info and not ready_info.get("ready"):
            st.warning(f"Up, but not fully ready: {ready_info.get('checks')}")
        else:
            st.success("Backend healthy and ready.")

    st.divider()
    st.session_state.show_debug = st.toggle(
        "Show technical details",
        value=st.session_state.show_debug,
        help="Reveal guardrail reasoning and retrieval metadata under each answer.",
    )
    st.caption("Off by default for a cleaner chat experience.")

# ----------------------------------------------------
# Main area: tabs
# ----------------------------------------------------
chat_tab, documents_tab = st.tabs(["💬 Chat", "📁 Documents"])

# --- Documents tab: upload + manage ---
with documents_tab:
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("PDF document", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None and st.button("Ingest Document", type="primary"):
        with st.spinner("Indexing vector embeddings and extracting knowledge graph..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/upload",
                    files=files,
                    headers=_headers(),
                    timeout=UPLOAD_TIMEOUT,
                )
                if response.status_code == 202:
                    st.success(
                        f"Queued '{uploaded_file.name}' for ingestion. "
                        "Use Refresh below to check its status."
                    )
                    st.session_state.documents = None  # force refresh below
                else:
                    st.error(f"Ingestion failed ({response.status_code}): {_extract_error_detail(response)}")
            except requests.exceptions.Timeout:
                st.error(
                    f"Upload timed out after {UPLOAD_TIMEOUT:g} seconds. "
                    "Ingestion is still synchronous and may take longer for documents "
                    "with many pages. Check the backend logs before retrying."
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend to upload the document.")
            except requests.exceptions.RequestException as e:
                st.error(f"Unexpected error during upload: {str(e)}")

    st.divider()
    st.subheader("Ingested documents")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        refresh_clicked = st.button("🔄 Refresh", use_container_width=True)
    with col_b:
        clear_clicked = st.button("Clear list", use_container_width=True)

    if clear_clicked:
        st.session_state.documents = None

    if refresh_clicked or st.session_state.documents is None:
        with st.spinner("Loading documents..."):
            ok, err = fetch_documents()
            if not ok:
                st.error(err)

    documents = st.session_state.documents or []
    if not documents:
        st.caption("No documents ingested yet.")
    else:
        for doc in documents:
            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    st.write(f"**{doc.get('filename', 'unknown')}**")
                    st.caption(
                        f"Status: `{doc.get('status')}` · "
                        f"Chunks: {doc.get('chunks_indexed', 0)} · "
                        f"Graph docs: {doc.get('graph_documents_extracted', 0)} · "
                        f"Uploaded: {doc.get('created_at', '-')[:19].replace('T', ' ')}"
                    )
                    if doc.get("error"):
                        st.caption(f"⚠️ {doc['error']}")
                with right:
                    if st.button("🗑️", key=f"delete_{doc.get('doc_id')}", help="Delete this document"):
                        with st.spinner("Deleting..."):
                            try:
                                del_resp = requests.delete(
                                    f"{BACKEND_URL}/api/v1/documents/{doc.get('doc_id')}",
                                    headers=_headers(),
                                    timeout=DOCS_TIMEOUT,
                                )
                                if del_resp.status_code == 200:
                                    fetch_documents()
                                    st.rerun()
                                else:
                                    st.error(f"Delete failed: {_extract_error_detail(del_resp)}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Delete error: {str(e)}")

# --- Chat tab ---
with chat_tab:
    if not st.session_state.messages:
        st.info("Ask a question about your uploaded documents to get started.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            metadata = msg.get("metadata")
            if metadata:
                render_guardrail_badge(metadata)
                if st.session_state.show_debug:
                    render_guardrail_details(metadata)

    if user_query := st.chat_input("Ask a question about your uploaded document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/api/v1/query",
                        json={"query": user_query},
                        headers=_headers(),
                        timeout=QUERY_TIMEOUT,
                    )

                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "")
                        meta = {
                            "status": data.get("guardrail_status", "UNKNOWN"),
                            "reason": data.get("reason"),
                            "retrieval_metadata": data.get("retrieval_metadata"),
                        }

                        st.markdown(answer)
                        render_guardrail_badge(meta)
                        if st.session_state.show_debug:
                            render_guardrail_details(meta)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "metadata": meta}
                        )
                    else:
                        err_msg = _extract_error_detail(res)
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})

                except requests.exceptions.Timeout:
                    err_msg = "The query timed out. Please try again."
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                except requests.exceptions.ConnectionError:
                    err_msg = "Could not connect to the backend. Ensure it is running and reachable."
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                except requests.exceptions.RequestException as e:
                    err_msg = f"Unexpected error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
