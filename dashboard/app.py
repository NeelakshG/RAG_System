import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from dashboard.components import STYLE, render_badge, render_chunk_card, render_meter, render_strategy_chip

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
COMPARISON_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "comparison.json"

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

st.title("RAG Pipeline")
st.caption("Hybrid search, grounded generation, verified citations — all local.")

tab_ask, tab_documents, tab_eval = st.tabs(["Ask", "Documents", "Eval comparison"])

with tab_ask:
    with st.form("ask_form"):
        question = st.text_input("Question", placeholder="What does ERR_2043 indicate?")
        use_hybrid = st.toggle("Hybrid search (dense + sparse)", value=True)
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Retrieving, generating, and verifying citations..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/v1/ask",
                    json={"question": question, "use_hybrid": use_hybrid},
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
                    f'<span style="font-size:0.85rem;">[{c["citation_number"]}] {html.escape(c["claim"])}</span>'
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

    with st.expander("Re-ingest corpus"):
        corpus_dir = st.text_input("Corpus directory", value="data/corpus")
        if st.button("Ingest"):
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
