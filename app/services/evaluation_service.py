"""Offline retrieval benchmarking for DocuMind AI.

The benchmark uses a frozen, versioned corpus snapshot so CI results are
reproducible and do not depend on a developer's local ChromaDB directory.
No Gemini calls are made: this suite measures retrieval quality, not answer
quality. expected_answer is retained as the human-reviewed answer key.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.services.embedding_service import embed_texts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "questions_with_chunks.json"
DEFAULT_CORPUS = PROJECT_ROOT / "evaluation" / "corpus_snapshot.json"
CROSS_ENCODER_NAME = os.getenv(
    "RAG_EVAL_CROSS_ENCODER",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
METHODS = (
    "simple_chunking",
    "semantic_chunking",
    "hybrid_search",
    "hybrid_reranking",
)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)



def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    questions = _read_json(path)

    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must be a JSON array.")

    if len(questions) != 50:
        raise ValueError(
            f"Expected exactly 50 evaluation questions; found {len(questions)}."
        )

    required = {
        "id",
        "question",
        "expected_answer",
        "source",
        "relevant_pages",
        "relevant_chunk_ids",
        "relevance_status",
    }

    seen: set[str] = set()

    for index, item in enumerate(questions, start=1):
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"Question {index} is missing fields: {sorted(missing)}"
            )

        if not item["question"].strip() or not item["expected_answer"].strip():
            raise ValueError(
                f"Question {index} must have a question and expected_answer."
            )

        if not item["relevant_pages"] or any(
            int(page) < 1 for page in item["relevant_pages"]
        ):
            raise ValueError(
                f"Question {index} must define at least one positive relevant page."
            )

        chunk_ids = item["relevant_chunk_ids"]

        if not isinstance(chunk_ids, list) or not chunk_ids:
            raise ValueError(
                f"Question {index} must define at least one relevant chunk ID."
            )

        if any(
            not isinstance(chunk_id, int)
            or isinstance(chunk_id, bool)
            or chunk_id < 0
            for chunk_id in chunk_ids
        ):
            raise ValueError(
                f"Question {index} contains invalid relevant chunk IDs."
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                f"Question {index} contains duplicate relevant chunk IDs."
            )

        if item["id"] in seen:
            raise ValueError(
                f"Duplicate evaluation question id: {item['id']}"
            )

        seen.add(item["id"])

    return questions


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict]:
    records = _read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("Evaluation corpus snapshot is empty or invalid.")
    required = {"id", "text", "document_id", "source", "page_number", "chunk_id"}
    for index, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Corpus record {index} is missing fields: {sorted(missing)}")
    return records


def _normalise_vectors(vectors: list[list[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Embedding model must return a 2D vector matrix.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25:
    """Small dependency-free BM25 implementation for the evaluation corpus."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [_tokenize(text) for text in documents]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.tokens:
            self.doc_freq.update(set(tokens))
        self.lengths = [len(tokens) for tokens in self.tokens]

    def scores(self, query: str) -> list[float]:
        query_terms = _tokenize(query)
        total_docs = len(self.tokens)
        scores = [0.0] * total_docs
        for term in query_terms:
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for index, tokens in enumerate(self.tokens):
                frequency = tokens.count(term)
                if not frequency:
                    continue
                length_norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1))
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + length_norm)
        return scores


def _semantic_chunks(corpus: list[dict], max_chars: int = 900, similarity_floor: float = 0.62) -> list[dict]:
    """Re-segment each source page at semantic sentence boundaries."""
    pages: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for record in corpus:
        key = (record["document_id"], record["source"], int(record["page_number"]))
        pages[key].append(record)

    output: list[dict] = []
    for (document_id, source, page_number), records in pages.items():
        records.sort(key=lambda item: int(item.get("start_index", 0)))
        # Reconstruct original page text from overlapping fixed-size chunks
        # using their original character offsets, avoiding duplicated overlap.
        page_chars: list[str] = []
        for record in records:
            start = max(0, int(record.get("start_index", 0)))
            chunk_text = record["text"]
            required_length = start + len(chunk_text)
            if len(page_chars) < required_length:
                page_chars.extend([" "] * (required_length - len(page_chars)))
            for offset, character in enumerate(chunk_text):
                position = start + offset
                if page_chars[position] == " ":
                    page_chars[position] = character
        page_text = "".join(page_chars).strip()
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", page_text) if part.strip()]
        if not sentences:
            continue
        sentence_vectors = _normalise_vectors(embed_texts(sentences))
        current: list[str] = []
        current_vectors: list[np.ndarray] = []

        def flush() -> None:
            if current:
                output.append({
                    "id": f"semantic_{len(output)}",
                    "text": " ".join(current),
                    "document_id": document_id,
                    "source": source,
                    "page_number": page_number,
                    "chunk_id": len(output),
                })

        for sentence, vector in zip(sentences, sentence_vectors):
            if current_vectors:
                centroid = np.mean(current_vectors, axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                similarity = float(np.dot(centroid, vector))
                current_length = sum(len(part) for part in current)
                if similarity < similarity_floor or current_length + len(sentence) > max_chars:
                    flush()
                    current, current_vectors = [], []
            current.append(sentence)
            current_vectors.append(vector)
        flush()
    return output


@lru_cache(maxsize=1)
def _get_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("Install sentence-transformers to run the reranking benchmark.") from exc
    return CrossEncoder(CROSS_ENCODER_NAME)


def _rank_indices(query_vector: np.ndarray, matrix: np.ndarray) -> list[int]:
    scores = matrix @ query_vector
    return np.argsort(-scores).tolist()


def _rrf(dense_ranked: list[int], sparse_ranked: list[int], limit: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in (dense_ranked, sparse_ranked):
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (60 + rank)
    return sorted(scores, key=scores.get, reverse=True)[:limit]



def _retrieval_metrics(
    results: list[dict],
    question: dict,
    top_k: int,
    evaluate_chunks: bool = True,
) -> dict:
    """Calculate page-level metrics and, when comparable, chunk-level metrics."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    relevant_pages = {int(page) for page in question["relevant_pages"]}
    relevant_source = question["source"]
    relevant_document = question.get("document_id")
    top_results = results[:top_k]

    relevant_page_ranks = [
        rank
        for rank, result in enumerate(top_results, start=1)
        if result["source"] == relevant_source
        and int(result["page_number"]) in relevant_pages
        and (
            not relevant_document
            or result.get("document_id") == relevant_document
        )
    ]
    found_pages = {
        int(result["page_number"])
        for result in top_results
        if result["source"] == relevant_source
        and int(result["page_number"]) in relevant_pages
        and (
            not relevant_document
            or result.get("document_id") == relevant_document
        )
    }

    metrics = {
        "hit_at_k": 1.0 if relevant_page_ranks else 0.0,
        "precision_at_k": len(relevant_page_ranks) / top_k,
        "recall_at_k": len(found_pages) / len(relevant_pages),
        "mrr": 1.0 / relevant_page_ranks[0] if relevant_page_ranks else 0.0,
    }

    # Semantic chunks have newly generated IDs, so original chunk labels
    # cannot be compared to them. Keep their chunk metrics explicitly null.
    if not evaluate_chunks or "relevant_chunk_ids" not in question:
        metrics.update({
            "chunk_hit_at_k": None,
            "chunk_precision_at_k": None,
            "chunk_recall_at_k": None,
            "chunk_mrr": None,
        })
        return metrics

    relevant_chunk_ids = {
        int(chunk_id) for chunk_id in question["relevant_chunk_ids"]
    }
    relevant_chunk_ranks = [
        rank
        for rank, result in enumerate(top_results, start=1)
        if result.get("chunk_id") is not None
        and int(result["chunk_id"]) in relevant_chunk_ids
        and result["source"] == relevant_source
        and (
            not relevant_document
            or result.get("document_id") == relevant_document
        )
    ]
    found_chunk_ids = {
        int(result["chunk_id"])
        for result in top_results
        if result.get("chunk_id") is not None
        and int(result["chunk_id"]) in relevant_chunk_ids
        and result["source"] == relevant_source
        and (
            not relevant_document
            or result.get("document_id") == relevant_document
        )
    }

    metrics.update({
        "chunk_hit_at_k": 1.0 if relevant_chunk_ranks else 0.0,
        "chunk_precision_at_k": len(relevant_chunk_ranks) / top_k,
        "chunk_recall_at_k": (
            len(found_chunk_ids) / len(relevant_chunk_ids)
            if relevant_chunk_ids else 0.0
        ),
        "chunk_mrr": (
            1.0 / relevant_chunk_ranks[0]
            if relevant_chunk_ranks else 0.0
        ),
    })
    return metrics


def run_benchmark(
    dataset_path: str | Path = DEFAULT_DATASET,
    corpus_path: str | Path = DEFAULT_CORPUS,
    output_path: str | Path | None = None,
    top_k: int = 5,
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    questions = load_dataset(dataset_path)
    corpus = load_corpus(corpus_path)
    texts = [record["text"] for record in corpus]

    dense_matrix = _normalise_vectors(embed_texts(texts))
    bm25 = _BM25(texts)
    semantic_corpus = _semantic_chunks(corpus)
    semantic_matrix = _normalise_vectors(
        embed_texts([item["text"] for item in semantic_corpus])
    )

    reranker = _get_cross_encoder()
    # Exclude model loading/warm-up from per-query latency.
    reranker.predict(
        [("warmup query", "warmup document")],
        show_progress_bar=False,
    )

    summaries: dict[str, dict[str, list[float]]] = {}
    question_results: list[dict] = []
    for method in METHODS:
        summaries[method] = {
            "hit_at_k": [],
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": [],
            "chunk_hit_at_k": [],
            "chunk_precision_at_k": [],
            "chunk_recall_at_k": [],
            "chunk_mrr": [],
            "latency_ms": [],
        }

    for question in questions:
        query_vector = _normalise_vectors(
            embed_texts([question["question"]])
        )[0]
        allowed_doc = question.get("document_id")

        for method in METHODS:
            method_started = time.perf_counter()

            if method == "simple_chunking":
                eligible = [
                    i for i, item in enumerate(corpus)
                    if not allowed_doc or item["document_id"] == allowed_doc
                ]
                order = sorted(
                    eligible,
                    key=lambda i: float(np.dot(dense_matrix[i], query_vector)),
                    reverse=True,
                )[:top_k]
                selected = [corpus[i] for i in order]

            elif method == "semantic_chunking":
                eligible = [
                    i for i, item in enumerate(semantic_corpus)
                    if not allowed_doc or item["document_id"] == allowed_doc
                ]
                order = sorted(
                    eligible,
                    key=lambda i: float(np.dot(semantic_matrix[i], query_vector)),
                    reverse=True,
                )[:top_k]
                selected = [semantic_corpus[i] for i in order]

            elif method == "hybrid_search":
                eligible = [
                    i for i, item in enumerate(corpus)
                    if not allowed_doc or item["document_id"] == allowed_doc
                ]
                dense_order = sorted(
                    eligible,
                    key=lambda i: float(np.dot(dense_matrix[i], query_vector)),
                    reverse=True,
                )
                sparse_scores = bm25.scores(question["question"])
                sparse_order = sorted(
                    eligible,
                    key=lambda i: sparse_scores[i],
                    reverse=True,
                )
                order = _rrf(dense_order, sparse_order, top_k)
                selected = [corpus[i] for i in order]

            else:  # hybrid_reranking: dense + BM25 -> RRF -> Cross-Encoder
                eligible = [
                    i for i, item in enumerate(corpus)
                    if not allowed_doc or item["document_id"] == allowed_doc
                ]
                dense_order = sorted(
                    eligible,
                    key=lambda i: float(np.dot(dense_matrix[i], query_vector)),
                    reverse=True,
                )
                sparse_scores = bm25.scores(question["question"])
                sparse_order = sorted(
                    eligible,
                    key=lambda i: sparse_scores[i],
                    reverse=True,
                )

                candidate_indices = _rrf(
                    dense_order,
                    sparse_order,
                    limit=min(20, len(eligible)),
                )
                candidates = [corpus[i] for i in candidate_indices]

                if candidates:
                    pairs = [
                        (question["question"], item["text"])
                        for item in candidates
                    ]
                    scores = reranker.predict(
                        pairs,
                        show_progress_bar=False,
                    )
                    ranked_candidates = sorted(
                        zip(scores, candidates),
                        key=lambda pair: float(pair[0]),
                        reverse=True,
                    )
                    selected = [
                        item for _, item in ranked_candidates[:top_k]
                    ]
                else:
                    selected = []

            metrics = _retrieval_metrics(
                selected,
                question,
                top_k,
                evaluate_chunks=(method != "semantic_chunking"),
            )
            latency = (time.perf_counter() - method_started) * 1000

            for key, value in metrics.items():
                if value is not None:
                    summaries[method][key].append(value)
            summaries[method]["latency_ms"].append(latency)

            question_results.append({
                "question_id": question["id"],
                "method": method,
                **metrics,
                "latency_ms": round(latency, 3),
                "retrieved": [
                    {
                        "source": item["source"],
                        "document_id": item.get("document_id"),
                        "page_number": int(item["page_number"]),
                        "chunk_id": item.get("chunk_id"),
                        "text": item["text"][:300],
                    }
                    for item in selected
                ],
            })

    aggregate = {}
    for method, values in summaries.items():
        aggregate[method] = {
            "accuracy_hit_at_k": round(
                statistics.mean(values["hit_at_k"]), 4
            ),
            "precision_at_k": round(
                statistics.mean(values["precision_at_k"]), 4
            ),
            "recall_at_k": round(
                statistics.mean(values["recall_at_k"]), 4
            ),
            "mrr": round(statistics.mean(values["mrr"]), 4),
            "chunk_hit_at_k": (
                round(statistics.mean(values["chunk_hit_at_k"]), 4)
                if values["chunk_hit_at_k"] else None
            ),
            "chunk_precision_at_k": (
                round(statistics.mean(values["chunk_precision_at_k"]), 4)
                if values["chunk_precision_at_k"] else None
            ),
            "chunk_recall_at_k": (
                round(statistics.mean(values["chunk_recall_at_k"]), 4)
                if values["chunk_recall_at_k"] else None
            ),
            "chunk_mrr": (
                round(statistics.mean(values["chunk_mrr"]), 4)
                if values["chunk_mrr"] else None
            ),
            "average_latency_ms": round(
                statistics.mean(values["latency_ms"]), 3
            ),
            "questions_evaluated": len(values["hit_at_k"]),
            "chunk_questions_evaluated": len(values["chunk_hit_at_k"]),
        }

    report = {
        "project": "DocuMind AI",
        "benchmark_version": 2,
        "dataset_questions": len(questions),
        "corpus_chunks": len(corpus),
        "semantic_chunks": len(semantic_corpus),
        "top_k": top_k,
        "metric_definition": (
            "Page-level metrics use relevant source-page labels. "
            "Chunk-level metrics use relevant_chunk_ids from the dataset. "
            "Semantic chunking is not scored against original fixed-size "
            "chunk IDs because its chunk boundaries and IDs differ."
        ),
        "chunk_label_status": "provisional_keyword_based",
        "answer_quality_note": (
            "This is an offline retrieval benchmark. It does not call Gemini "
            "or measure generated-answer correctness or faithfulness. "
            "Chunk relevance labels are provisional keyword-based labels."
        ),
        "methods": aggregate,
        "question_results": question_results,
    }

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report
