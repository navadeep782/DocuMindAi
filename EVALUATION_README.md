# KnowledgeHub AI — RAG Retrieval Evaluation

This adds the three-stage workflow from the project plan:

1. A fixed 50-question ground-truth set with reviewed expected answers and relevant source pages.
2. A controlled comparison of four retrieval strategies: existing fixed-size/simple chunks, semantic chunking, hybrid dense + BM25 retrieval using Reciprocal Rank Fusion, and dense retrieval followed by a Cross-Encoder reranker.
3. A GitHub Actions workflow that runs the benchmark and enforces a configurable retrieval hit-rate quality gate.

## Important evaluation definition

`accuracy_hit_at_k` is retrieval hit rate: the fraction of questions for which at least one retrieved result matches a labelled relevant source page. The benchmark also reports Precision@K, Recall@K, MRR and average retrieval latency. It does **not** call Gemini and does not claim to measure generated-answer correctness or faithfulness. `expected_answer` is the answer key for human review and for a future answer-generation evaluation stage.

The included `evaluation/corpus_snapshot.json` is a frozen export of the uploaded ChromaDB snapshot (390 chunks from `DBMS_Full_Notes.pdf`, document ID `doc_000ea43f4ea64a758b7960eb340793b7`). This makes GitHub Actions reproducible without committing the local ChromaDB binary database. If the source corpus changes, regenerate this snapshot and update the dataset labels.

## Run locally

From the repository root, activate your project virtual environment and install the benchmark dependencies:

```bash
pip install -r requirements-evaluation.txt
python scripts/run_rag_evaluation.py --top-k 5 --min-hit-rate 0.50
```

The first run downloads the embedding model and the Cross-Encoder model. The output report is written to `evaluation/reports/latest.json` (reports are git-ignored by default only if you add that rule; choose whether to commit selected reports).

To change the quality gate:

```bash
python scripts/run_rag_evaluation.py --min-hit-rate 0.60
```

## How the methods differ

- **Simple chunking:** dense cosine-similarity search over the existing stored chunks.
- **Semantic chunking:** rebuilds page-level text from the snapshot and groups adjacent sentences using embedding similarity, with a maximum chunk length.
- **Hybrid search:** combines dense vector ranks and BM25 keyword ranks using Reciprocal Rank Fusion.
- **Reranking:** retrieves dense candidates, then reorders them with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

All methods use the same questions, corpus, document filter, relevance labels and K value. The reranker model and embedding model are loaded from Sentence Transformers; first execution requires model download access.

## Dataset maintenance

The 50 answer keys and relevant page labels are a starting benchmark curated against the supplied DBMS notes snapshot. Review them against the original PDF before presenting benchmark scores publicly. Add questions only when their expected answer and relevant source page(s) have been independently checked. Keep the dataset at 50 items unless you intentionally update the workflow's exact-count validation.

## CI quality gate

`.github/workflows/rag-evaluation.yml` runs when backend service code, evaluation assets, or the workflow changes. It uploads the JSON report as a GitHub Actions artifact, including when the quality gate fails. The initial threshold is 0.50 hit@5 for every method; after observing repeatable baseline results, raise this threshold based on measured performance rather than guessing.

## Git ignore recommendation

Append these patterns to your existing `.gitignore` (do not replace an existing ignore file):

```gitignore
/evaluation/reports/*.json
__pycache__/
*.py[cod]
.pytest_cache/
.env
.venv/
/chroma_db/
```
