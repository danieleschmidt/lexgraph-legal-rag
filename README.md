# LexGraph Legal RAG

**FAISS-based multi-agent legal document retrieval system — DocGraph's retrieval layer.**

Index legal documents, retrieve relevant clauses with citations, and feed results into contract extractors and law provers.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage: 93%](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](#tests)

---

## Overview

LexGraph Legal RAG is a lightweight, offline legal document retrieval system built on:

- **FAISS** — efficient nearest-neighbor search over document embeddings
- **TF-IDF (char-trigram)** — document embeddings with no external API calls
- **Multi-agent retrieval** — parallel query expansion for improved recall
- **Citation-aware results** — every result includes doc name, section, and char offset

It is designed to serve as **DocGraph's document retrieval layer**, feeding into:
- [`multimodal-contract-extractor`](https://github.com/danieleschmidt/multimodal-contract-extractor) — receives top-k chunks to extract structured contract data
- [`Neuro-Symbolic-Law-Prover`](https://github.com/danieleschmidt/Neuro-Symbolic-Law-Prover) — receives retrieved clauses + citations for logical reasoning over legal rules

---

## Architecture

```
Legal Documents
      │
      ▼
LegalDocumentIndexer
  ├── splits by section/clause (regex-based)
  ├── embeds with TF-IDF char-trigrams (8192 features)
  └── stores normalized vectors in FAISS IndexFlatIP
      │
      ▼
LegalRetriever
  ├── embeds query with same vectorizer
  ├── runs FAISS inner-product search
  └── returns RetrievedChunk list with citations
      │
      ▼
MultiAgentRetriever
  ├── expands query into 2–4 variants
  ├── runs all variants in parallel (ThreadPoolExecutor)
  ├── deduplicates by (doc_name, section)
  └── returns merged ranked list
      │
      ▼
DocGraph Integration
  ├── → multimodal-contract-extractor
  └── → Neuro-Symbolic-Law-Prover
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/danieleschmidt/lexgraph-legal-rag.git
cd lexgraph-legal-rag
pip install -e .

# Run the demo
python demo.py
```

### Code Example

```python
from lexgraph_legal_rag.indexer import LegalDocumentIndexer
from lexgraph_legal_rag.retriever import LegalRetriever, MultiAgentRetriever

# 1. Index documents
indexer = LegalDocumentIndexer(chunk_size=512, chunk_overlap=64)
indexer.index_documents({
    "NDA_2024": open("nda.txt").read(),
    "Software_License": open("license.txt").read(),
})

# 2. Single retriever
retriever = LegalRetriever(indexer=indexer, top_k=5)
results = retriever.retrieve("What are the confidentiality obligations?")

for r in results:
    print(f"Score: {r.score:.4f} | {r.citation}")
    print(f"  {r.text[:200]}...")

# 3. Multi-agent retriever (expands query, deduplicates)
mar = MultiAgentRetriever(indexer=indexer, top_k=5)
results = mar.retrieve("software warranty disclaimer limitations")

# 4. Persist index
indexer.save("legal_index.faiss")

# 5. Load index
indexer2 = LegalDocumentIndexer()
indexer2.load("legal_index.faiss")
```

---

## Core Classes

### `LegalDocumentIndexer`

Splits legal documents by section/clause headers and stores embeddings in FAISS.

```python
indexer = LegalDocumentIndexer(
    chunk_size=512,       # max chars per chunk
    chunk_overlap=64,     # overlap between sub-chunks
)
indexer.index_documents({"doc_name": "...full text..."})
```

**Chunking strategy:**
1. Finds section boundaries via regex (`Section X.`, `Article X.`, `Clause X.`, ALL CAPS headings, `WHEREAS`, etc.)
2. If a section exceeds `chunk_size`, splits further with overlap
3. Returns `DocumentChunk` objects with `doc_name`, `section`, `text`, `char_offset`

### `LegalRetriever`

Queries the FAISS index and returns ranked results with citations.

```python
retriever = LegalRetriever(indexer=indexer, top_k=5, min_score=0.0)
results = retriever.retrieve("indemnification clause")
# → List[RetrievedChunk], sorted by score descending
```

Each `RetrievedChunk` has:
- `.score` — cosine similarity score (0–1)
- `.citation` — `"DocName § Section (offset N)"`
- `.text` — the chunk text
- `.chunk.doc_name`, `.chunk.section`, `.chunk.char_offset`

### `MultiAgentRetriever`

Expands a query into variants, runs them in parallel, and merges results.

```python
mar = MultiAgentRetriever(indexer=indexer, top_k=5, max_workers=4)
results = mar.retrieve("liability for damages under contract")
```

Query expansion strategies:
- Strip question words → keyword form
- Add "legal" qualifier
- Extract capitalized noun phrases

---

## DocGraph Integration

### → multimodal-contract-extractor

```python
from lexgraph_legal_rag.retriever import MultiAgentRetriever

# Feed retrieved chunks as context for contract extraction
retriever = MultiAgentRetriever(indexer=indexer)
chunks = retriever.retrieve("payment terms and obligations")

# Pass chunk texts + citations to extractor
context = [(r.text, r.citation) for r in chunks]
# → extractor.extract(context, schema=ContractSchema)
```

### → Neuro-Symbolic-Law-Prover

```python
# Retrieved chunks provide the factual premises for logical reasoning
chunks = retriever.retrieve("termination for breach")

# Each chunk becomes a grounded premise with citation evidence
premises = [
    {"text": r.text, "source": r.citation, "score": r.score}
    for r in chunks
]
# → prover.prove(premises, query="Can contract be terminated immediately?")
```

The retriever's citation metadata (doc name, section, char offset) enables the law prover to trace every logical inference back to the original source document.

---

## Tests

```bash
# Install dev deps
pip install pytest pytest-cov

# Run tests
pytest tests/test_indexer.py tests/test_retriever.py -v

# Run all tests
pytest tests/ -v
```

Coverage: **93%** on core modules (`indexer.py`, `retriever.py`, `models.py`).

---

## Dependencies

- `faiss-cpu >= 1.7.0` — vector similarity search
- `scikit-learn >= 1.3.0` — TF-IDF vectorizer
- `numpy >= 1.24.0` — vector operations
- No OpenAI / external API calls required

---

## License

MIT — see [LICENSE](LICENSE)
