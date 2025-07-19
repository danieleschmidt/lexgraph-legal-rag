import streamlit as st
from lexgraph_legal_rag.context_reasoning import ContextAwareReasoner
from lexgraph_legal_rag.logging_config import configure_logging
from lexgraph_legal_rag.config import validate_environment

configure_logging()

# Validate configuration at startup
try:
    config = validate_environment()
    st.success("Configuration validated successfully")
except Exception as e:
    st.error(f"Configuration validation failed: {e}")
    st.stop()

st.title("LexGraph Legal RAG")
index_path = st.text_input("Index path", "index.bin")
query = st.text_area("Query")

if st.button("Search") and query:
    reasoner = ContextAwareReasoner()
    reasoner.pipeline.load_index(index_path)
    results = reasoner.run(query)
    st.write(results)
