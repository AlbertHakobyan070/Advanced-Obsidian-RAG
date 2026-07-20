"""
app.py — Streamlit interface for Personal RAG.

    streamlit run app.py      (or: python main.py serve)

Shows the answer, a confidence badge, expandable source excerpts with citation
audit marks, and a sidebar with pipeline config + timing. Built to be the thing
you actually open before an interview to drill your own notes.

Sidebar retrieval controls: pick a preset (auto/code/concept/synthesis) and/or
override top-k per query — the warm pipeline never restarts. "Save as default"
writes the value back to config.yaml (comment-preserving) so it survives
restarts too.
"""
from __future__ import annotations

import time

import streamlit as st

from src.pipeline import RAGPipeline
from src.utils.config_loader import load_config, persist_config_values

st.set_page_config(page_title="Personal RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading pipeline…")
def get_pipeline():
    cfg = load_config()
    return RAGPipeline.from_config(cfg), cfg


_CONF_COLOR = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴", "UNKNOWN": "⚪", "ERROR": "⚫"}


def main():
    rag, cfg = get_pipeline()

    with st.sidebar:
        st.header("⚙️ Pipeline")
        st.markdown(f"**Generation:** `{cfg.get('generation.provider')}`")
        st.markdown(f"**Model:** `{cfg.get('generation.model') if cfg.get('generation.provider') != 'local' else cfg.get('generation.local.model')}`")
        st.markdown(f"**Embedding:** `{cfg.get('embedding.provider')}`")
        st.markdown(f"**HyDE:** `{cfg.get('retrieval.use_hyde')}`")
        st.markdown(f"**Reranker:** `{cfg.get('retrieval.rerank_mode')}`")
        st.divider()

        st.subheader("🎛 Retrieval")
        preset_names = ["auto"] + sorted(rag.presets)
        preset = st.selectbox(
            "Preset", preset_names,
            help="auto = code preset kicks in on code-intent queries; "
                 "others force the bundle from config.yaml retrieval.presets.",
        )
        override_k = st.checkbox(
            "Override top-k", value=False,
            help="Per-query override of rerank_top_k (chunks reaching the LLM). "
                 "Wins over the preset's value.",
        )
        top_k = st.number_input(
            "top_k", min_value=1, max_value=50,
            value=int(rag.rerank_top_k), step=1,
            disabled=not override_k,
        )
        if st.button("💾 Save top-k as default",
                     help="Updates the warm pipeline AND config.yaml, so the "
                          "value survives restarts."):
            try:
                persist_config_values(cfg.project_root / "config.yaml",
                                      {"rerank_top_k": int(top_k)})
                rag.rerank_top_k = int(top_k)
                st.success(f"rerank_top_k = {int(top_k)} saved to config.yaml")
            except ValueError as e:
                st.error(str(e))
        st.divider()
        st.caption("Answers are grounded in your own lecture notes. "
                   "Citations are audited by a second model pass.")

    st.title("📚 the personal RAG")
    st.caption("Ask anything from your Obsidian vault — grounded, cited answers from your own notes.")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Render history
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            _render_answer(entry["answer"], entry["elapsed"])

    question = st.chat_input("Ask a question about your notes…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning…"):
                t0 = time.time()
                try:
                    answer = rag.query(
                        question,
                        preset=None if preset == "auto" else preset,
                        top_k=int(top_k) if override_k else None,
                    )
                except Exception as e:
                    answer = None
                    st.error(
                        "Generation backend unreachable — is the LLM proxy "
                        f"running? ({type(e).__name__}: {e})"
                    )
                elapsed = time.time() - t0
            if answer is not None:
                _render_answer(answer, elapsed)
        if answer is not None:
            st.session_state.history.append(
                {"question": question, "answer": answer, "elapsed": elapsed}
            )


def _render_answer(answer, elapsed: float):
    badge = _CONF_COLOR.get(answer.confidence, "⚪")
    st.markdown(answer.text)
    meta_line = f"{badge} **Confidence:** {answer.confidence}  ·  ⏱ {elapsed:.1f}s"
    if answer.retrieval:
        r = answer.retrieval
        preset = r.get("preset") or "default"
        meta_line += (f"  ·  🎛 {preset} · k={r.get('rerank_top_k')}"
                      f" · hyde={'on' if r.get('hyde_used') else 'off'}")
        if r.get("scope"):
            meta_line += f"  ·  🧭 {', '.join(r['scope'])}"
    st.markdown(meta_line)

    if answer.verification:
        overall = answer.verification.get("overall", "N/A")
        if overall == "FLAG":
            st.warning("⚠️ Citation audit flagged at least one unsupported citation.")
        elif overall == "PASS":
            st.success("✓ All citations verified against sources.")

    if answer.sources:
        with st.expander(f"📄 Sources ({len(answer.sources)})"):
            cited_nums = {c.number for c in answer.citations}
            for i, doc in enumerate(answer.sources, 1):
                cite = next((c for c in answer.citations if c.number == i), None)
                mark = ""
                if cite and cite.supported is True:
                    mark = " ✓"
                elif cite and cite.supported is False:
                    mark = f" ⚠ ({cite.note})"
                used = " · cited" if i in cited_nums else ""
                st.markdown(f"**[{i}] {doc.source_label}**{mark}{used}")
                st.caption(doc.text[:400] + ("…" if len(doc.text) > 400 else ""))
                st.divider()


if __name__ == "__main__":
    main()
