# LexGraph-Legal-RAG

Build a LangGraph-powered multi-agent system that retrieves, reasons over, and cites legal clauses from large document stores.

## Features

- **Multi-Agent Architecture**: Recursive graph of specialized tools (retriever, summarizer, clause-explainer) that intelligently decide when to call one another
- **Legal Document Pipeline**: Vector-DB indexing pipeline for statutes, SEC filings, contracts, and case law
- **Citation-Rich Responses**: Streaming answers with precise clause-level citations and source references
- **Semantic Search**: Advanced retrieval using legal-domain embeddings
- **FAISS Index**: Optional scalable index for large document collections
- **Metrics & Structured Logging**: Monitor query latency and output JSON logs; optional Prometheus server
- **Context-Aware Reasoning**: Multi-hop reasoning across related legal concepts

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/yourusername/lexgraph-legal-rag.git
cd lexgraph-legal-rag
pip install -r requirements.txt

# Set the API key required for the service
export API_KEY=mysecret

# Install pre-commit hooks
pre-commit install && pre-commit run --all-files

# Index your legal document corpus
# (add `--semantic` to enable semantic search)
python ingest.py --docs ./corpus --index index.bin --semantic --metrics-port 8001

# Run interactive query session
python run_agent.py --query "What constitutes indemnification in commercial contracts?" --index index.bin --metrics-port 8002

# Start web interface
streamlit run streamlit_app.py
```

All API requests must include the `X-API-Key` header matching the value of
`API_KEY`.

The pipeline saves both vector and semantic indices. The semantic index
is stored alongside the main index with a `.sem` suffix and loads
automatically when present.

If you pass `--metrics-port`, a Prometheus exporter starts to report search
latency and request counts.

## Architecture

```
Query → Router Agent → Retriever Agent → Summarizer Agent → Citation Agent → Response
                   ↓
              Legal Knowledge Graph
```

## Configuration

Create `.env` file:
```env
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_pinecone_key
LEGAL_CORPUS_PATH=./data/legal_docs
```

## Roadmap

- [ ] Add semantic versioning on API endpoints
- [ ] Deploy demo UI with Streamlit Cloud
- [ ] Integrate with Westlaw/LexisNexis APIs
- [ ] Add multi-jurisdiction support
- [ ] Implement legal precedent tracking

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for research and educational purposes. Always consult qualified legal professionals for legal advice.
