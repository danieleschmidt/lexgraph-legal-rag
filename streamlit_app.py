import streamlit as st
from lexgraph_legal_rag.context_reasoning import ContextAwareReasoner
from lexgraph_legal_rag.logging_config import configure_logging

configure_logging()

st.title("LexGraph Legal RAG")
index_path = st.text_input("Index path", "index.bin")
query = st.text_area("Query")

if st.button("Search") and query:
    reasoner = ContextAwareReasoner()
    reasoner.pipeline.load_index(index_path)
    results = reasoner.run(query)
    st.write(results)
