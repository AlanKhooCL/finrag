import streamlit as st
import requests
import json
import time

# Config
API_URL = "http://127.0.0.1:8000"
API_KEY = "finrag-dev-2026"
HEADERS = {"X-API-Key": API_KEY}

# Page setup
st.set_page_config(
    page_title="FinRAG",
    page_icon="📄",
    layout="wide"
)

# Header
st.title("📄 FinRAG")
st.caption("Financial Document Intelligence — RAG-powered Q&A over MAS regulatory documents")

# Sidebar
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API Online")
        st.metric("Documents indexed", health["vector_store"]["document_count"])
        st.caption(f"LLM: {health['model']}")
        st.caption(f"Embeddings: {health['embeddings']}")
        st.caption(f"Reranker: {health['reranker']}")
    except:
        st.error("API Offline — start uvicorn first")

    st.divider()
    st.header("Ingest Documents")
    st.caption("Add PDFs to data/docs/ then click below")
    if st.button("Run Ingest", type="primary"):
        with st.spinner("Ingesting documents..."):
            try:
                r = requests.post(
                    f"{API_URL}/ingest",
                    headers=HEADERS,
                    timeout=300
                )
                result = r.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Ingested {result['total_chunks']} chunks from {result['files_processed']} files")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)

# Main query interface
st.header("Ask a Question")

# Example questions
st.caption("Try these:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Risk management practices for FMCs"):
        st.session_state.question = "What are the key risk management practices for fund management companies?"
with col2:
    if st.button("MAS governance requirements"):
        st.session_state.question = "What governance requirements does MAS set for financial institutions?"
with col3:
    if st.button("Technology risk guidelines"):
        st.session_state.question = "What are MAS guidelines on technology risk management?"

# Query input
question = st.text_input(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are MAS requirements for AI governance in financial services?"
)

if st.button("Ask FinRAG", type="primary") and question:
    with st.spinner("Retrieving and generating answer..."):
        start = time.time()
        try:
            r = requests.post(
                f"{API_URL}/query",
                headers=HEADERS,
                json={"question": question, "top_k": top_k},
                timeout=60
            )
            result = r.json()
            elapsed = round((time.time() - start) * 1000)

            # Answer
            st.header("Answer")
            st.markdown(result["answer"])

            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latency", f"{result['latency_ms']}ms")
            with col2:
                st.metric("Sources used", len(result["sources"]))
            with col3:
                avg_rerank = round(
                    sum(s.get("rerank_score", 0) for s in result["sources"]) /
                    max(len(result["sources"]), 1), 3
                )
                st.metric("Avg rerank score", avg_rerank)

            # Sources
            st.divider()
            st.subheader("Source Citations")
            for i, source in enumerate(result["sources"]):
                with st.expander(
                    f"Source {i+1} — {source['source']} "
                    f"(rerank: {source.get('rerank_score', 'N/A')})"
                ):
                    st.caption(f"Chunk index: {source['chunk_index']} | "
                               f"Retrieval score: {source.get('retrieval_score', 'N/A')}")
                    st.write(source["text"])

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Eval scores footer
st.divider()
st.caption("📊 Evaluation scores — Avg relevancy: 0.724 | Avg rerank: 0.99 | Citation rate: 5/5 | Avg latency: 8.8s")