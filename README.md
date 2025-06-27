# LexGraph-Legal-RAG

Build a LangGraph-powered multi-agent system that retrieves, reasons over, and cites legal clauses from large document stores.

## Features

- **Multi-Agent Architecture**: Recursive graph of specialized tools (retriever, summarizer, clause-explainer) that intelligently decide when to call one another
- **Legal Document Pipeline**: Vector-DB indexing pipeline for statutes, SEC filings, contracts, and case law
- **Citation-Rich Responses**: Streaming answers with precise clause-level citations and source references
- **Semantic Search**: Advanced retrieval using legal-domain embeddings
- **Context-Aware Reasoning**: Multi-hop reasoning across related legal concepts

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/yourusername/lexgraph-legal-rag.git
cd lexgraph-legal-rag
pip install -r requirements.txt

# Index your legal document corpus
python ingest.py --docs ./corpus --index index.bin

# Run interactive query session
python run_agent.py --query "What constitutes indemnification in commercial contracts?" --index index.bin

# Start web interface
streamlit run app.py
```

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
