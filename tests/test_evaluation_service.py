import json

import pytest

from app.services import evaluation_service as evaluation


def test_dataset_contains_50_reviewed_questions():
    questions = evaluation.load_dataset()
    assert len(questions) == 50
    assert all(item["expected_answer"].strip() for item in questions)


def test_dataset_rejects_wrong_question_count(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 50"):
        evaluation.load_dataset(path)


def test_bm25_prefers_exact_keyword_match():
    index = evaluation._BM25([
        "database indexing speeds up search queries",
        "transactions provide atomicity and durability",
    ])
    scores = index.scores("database indexing")
    assert scores[0] > scores[1]


def test_retrieval_metrics_use_source_and_page_labels():
    question = {"source": "notes.pdf", "relevant_pages": [3, 4]}
    results = [
        {"source": "other.pdf", "page_number": 3},
        {"source": "notes.pdf", "page_number": 4},
    ]
    metrics = evaluation._retrieval_metrics(results, question, top_k=2)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr"] == 0.5


def test_corpus_snapshot_has_expected_records():
    corpus = evaluation.load_corpus()
    assert len(corpus) == 390
    assert all(item["text"].strip() for item in corpus)


def test_benchmark_runs_all_four_methods_with_deterministic_models(monkeypatch, tmp_path):
    import re
    import numpy as np

    def fake_embed_texts(texts):
        vectors = []
        for text in texts:
            vector = np.zeros(64, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                vector[hash(token) % 64] += 1.0
            if not np.any(vector):
                vector[0] = 1.0
            vectors.append(vector.tolist())
        return vectors

    class FakeCrossEncoder:
        def predict(self, pairs, show_progress_bar=False):
            scores = []
            for query, document in pairs:
                q = set(re.findall(r"[a-z0-9]+", query.lower()))
                d = set(re.findall(r"[a-z0-9]+", document.lower()))
                scores.append(len(q & d) / max(len(q), 1))
            return np.asarray(scores)

    monkeypatch.setattr(evaluation, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(evaluation, "_get_cross_encoder", lambda: FakeCrossEncoder())
    output = tmp_path / "benchmark.json"
    report = evaluation.run_benchmark(output_path=output, top_k=5)

    assert report["dataset_questions"] == 50
    assert set(report["methods"]) == set(evaluation.METHODS)
    assert all(value["questions_evaluated"] == 50 for value in report["methods"].values())
    assert output.exists()
