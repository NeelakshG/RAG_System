import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from dashboard.components import (
    STYLE,
    render_badge,
    render_chunk_card,
    render_meter,
    render_sources,
    render_strategy_chip,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
COMPARISON_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "comparison.json"

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

st.title("RAG Pipeline")
st.caption("Hybrid search, grounded generation, verified citations — all local.")

tab_ask, tab_documents, tab_eval = st.tabs(["Ask", "Documents", "Eval comparison"])

with tab_ask:
    scope = st.radio(
        "Search scope",
        ["All documents", "Specific documents"],
        horizontal=True,
        help="Restricting to specific documents skips embedding/reranking the rest of the "
        "corpus for this question -- faster, and avoids spending tokens on irrelevant context.",
    )

    selected_sources: list[str] = []
    if scope == "Specific documents":
        try:
            docs_response = requests.get(f"{API_BASE_URL}/v1/documents", timeout=30)
            docs_response.raise_for_status()
            doc_names = [d["source_name"] for d in docs_response.json()]
        except requests.RequestException as e:
            doc_names = []
            st.error(f"Couldn't load document list: {e}")
        selected_sources = st.multiselect("Documents to search", doc_names)

    with st.form("ask_form"):
        question = st.text_input("Question", placeholder="What does ERR_2043 indicate?")
        use_hybrid = st.toggle("Hybrid search (dense + sparse)", value=True)
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        if scope == "Specific documents" and not selected_sources:
            st.warning('Select at least one document, or switch to "All documents".')
        else:
            with st.spinner("Retrieving, generating, and verifying citations..."):
                try:
                    payload = {"question": question, "use_hybrid": use_hybrid}
                    if scope == "Specific documents":
                        payload["source_names"] = selected_sources
                    response = requests.post(
                        f"{API_BASE_URL}/v1/ask",
                        json=payload,
                        timeout=180,
                    )
                    response.raise_for_status()
                    st.session_state["last_result"] = response.json()
                except requests.RequestException as e:
                    st.error(f"Couldn't reach the API at {API_BASE_URL}: {e}")

    result = st.session_state.get("last_result")
    if result:
        answer_text = html.escape(result["answer"])
        fallback_class = " fallback" if result["fallback_triggered"] else ""
        st.markdown(f'<div class="hero-answer{fallback_class}">{answer_text}</div>', unsafe_allow_html=True)

        source_by_chunk_id = {chunk["chunk_id"]: chunk["source_name"] for chunk in result["chunks"]}
        sources_seen = list(dict.fromkeys(source_by_chunk_id.values()))
        if sources_seen:
            st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
            st.markdown(render_sources(sources_seen), unsafe_allow_html=True)

        col_confidence, col_citations = st.columns(2)

        with col_confidence:
            st.markdown('<div class="section-label">Confidence breakdown</div>', unsafe_allow_html=True)
            c = result["confidence"]
            meters = (
                render_meter("Retrieval", c["retrieval"])
                + render_meter("Citation coverage", c["citation"])
                + render_meter("Completeness", c["completeness"])
                + render_meter("Composite (gates fallback)", c["composite"], status_aware=True)
            )
            st.markdown(meters, unsafe_allow_html=True)

        with col_citations:
            st.markdown('<div class="section-label">Citations</div>', unsafe_allow_html=True)
            if result["citations"]:
                rows = "".join(
                    '<div style="margin-bottom:10px;">'
                    f'{render_badge(c["supported"])} '
                    f'<span style="font-size:0.85rem;">[{c["citation_number"]}] {html.escape(c["claim"])}</span> '
                    f'<span class="citation-source">— {html.escape(source_by_chunk_id.get(c["chunk_id"], "unknown"))}</span>'
                    "</div>"
                    for c in result["citations"]
                )
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span style="color:var(--text-muted); font-size:0.85rem;">No citations to verify.</span>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-label">Retrieved chunks</div>', unsafe_allow_html=True)
        for rank, chunk in enumerate(result["chunks"], start=1):
            st.markdown(
                render_chunk_card(
                    chunk["source_name"], chunk["section_heading"], chunk["score"], chunk["text"], rank
                ),
                unsafe_allow_html=True,
            )

with tab_documents:
    st.markdown('<div class="section-label">Indexed documents</div>', unsafe_allow_html=True)
    try:
        response = requests.get(f"{API_BASE_URL}/v1/documents", timeout=30)
        response.raise_for_status()
        docs = response.json()
        if docs:
            st.dataframe(docs, use_container_width=True, hide_index=True)
        else:
            st.info("No documents indexed yet. Ingest a corpus below to get started.")
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API at {API_BASE_URL}: {e}")

    with st.expander("Add documents", expanded=True):
        st.caption(
            "1. Choose one or more files.  2. Click **Upload & index**.  3. They'll show up in "
            "the table above once indexing finishes.\n\n"
            "Files are saved into `data/corpus/` and the whole corpus is then re-indexed — "
            "exact duplicates of chunks already indexed are skipped automatically, so "
            "re-uploading something is harmless."
        )
        uploaded_files = st.file_uploader(
            "md, txt, html, docx, or pdf",
            type=["md", "txt", "html", "docx", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Upload & index"):
            corpus_path = Path("data/corpus")
            corpus_path.mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_files:
                (corpus_path / uploaded_file.name).write_bytes(uploaded_file.getvalue())
            with st.spinner("Chunking, embedding, and indexing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/v1/ingest", json={"corpus_dir": str(corpus_path)}, timeout=300
                    )
                    response.raise_for_status()
                    stats = response.json()
                    st.success(
                        f"Uploaded {len(uploaded_files)} file(s), indexed {stats['indexed']} chunks "
                        f"({stats['deduped']} deduped as exact matches)."
                    )
                except requests.RequestException as e:
                    st.error(f"Ingestion failed: {e}")

    with st.expander("Advanced: re-index from a server-side folder"):
        st.caption(
            "**This box takes a folder path, not a file.** And that path is read on the "
            "computer running the API — which may not be the computer you're looking at this "
            "page on.\n\n"
            "- Running everything on one laptop (the normal setup here)? Then \"the API's "
            "computer\" is just your own machine, and a path like `data/corpus` or "
            "`C:/Users/you/Documents/reports` works as expected.\n"
            "- Running via Docker Compose? The API runs inside a container that can only see "
            "the `./data` folder on your machine (mounted in as `/app/data`). A path outside "
            "that — even one that exists on your own computer — will fail with a "
            "\"not found\" error, because the container never sees it.\n\n"
            "**What it actually does:** on click, it scans *every* supported file "
            "(.md, .txt, .html, .docx, .pdf) already sitting in that folder and re-embeds and "
            "re-indexes all of them — not just new ones. Files already indexed are cheap to "
            "redo (exact duplicates get skipped), but it's still reprocessing the whole folder "
            "each time, not appending one file.\n\n"
            "**If you just want to add a file from your own computer, don't use this** — use "
            "\"Add documents\" above. That one transfers the file through your browser to the "
            "server for you, so you never have to think about which machine sees which folder. "
            "This box exists only for files that got onto the server some other way (a script, "
            "a volume mount, manually copying them there) and now need to be picked up."
        )
        corpus_dir = st.text_input("Corpus directory", value="data/corpus")
        if st.button("Ingest folder"):
            with st.spinner("Chunking, embedding, and indexing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/v1/ingest", json={"corpus_dir": corpus_dir}, timeout=300
                    )
                    response.raise_for_status()
                    stats = response.json()
                    st.success(f"Indexed {stats['indexed']} chunks ({stats['deduped']} deduped).")
                except requests.RequestException as e:
                    st.error(f"Ingestion failed: {e}")

with tab_eval:
    st.markdown('<div class="section-label">Chunking strategy comparison</div>', unsafe_allow_html=True)
    if COMPARISON_PATH.exists():
        summaries = json.loads(COMPARISON_PATH.read_text())
        rows = []
        for s in summaries:
            overall = s["scores"]["overall"]
            rows.append(
                {
                    "strategy": s["strategy"],
                    "chunks": s["chunks_indexed"],
                    "deduped": s["chunks_deduped"],
                    "mean tokens/chunk": s["mean_chunk_tokens"],
                    "correctness": overall["correctness"],
                    "faithfulness": overall["faithfulness"],
                    "retrieval relevance": overall["retrieval_relevance"],
                    "citation accuracy": overall["citation_accuracy"],
                    "fallback rate": overall["fallback_rate"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        chips = "".join(
            f'<span style="margin-right:18px;">{render_strategy_chip(r["strategy"])}</span>' for r in rows
        )
        st.markdown(f'<div style="margin-top:8px;">{chips}</div>', unsafe_allow_html=True)
    else:
        st.info("No comparison results yet. Run `python scripts/run_eval.py` first.")
