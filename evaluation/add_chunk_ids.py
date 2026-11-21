
import json
from pathlib import Path

QUESTIONS = Path("evaluation/questions.json")
CANDIDATES = Path("evaluation/reports/chunk_candidates.json")
OUTPUT = Path("evaluation/questions_with_chunks.json")

with open(QUESTIONS) as f:
    questions = json.load(f)

with open(CANDIDATES) as f:
    candidate_data = json.load(f)

candidate_map = {
    item["question_id"]: item["candidates"]
    for item in candidate_data
}

updated = []
missing = []

for question in questions:
    candidates = candidate_map.get(question["id"], [])

    if not candidates:
        missing.append(question["id"])
        question_copy = question.copy()
        question_copy["relevant_chunk_ids"] = []
        question_copy["relevance_status"] = "unmapped"
    else:
        best_score = candidates[0]["score"]

        # Keep candidates scoring at least 80% of the best candidate.
        selected = [
            c for c in candidates
            if c["score"] >= best_score * 0.80
        ]

        question_copy = question.copy()
        question_copy["relevant_chunk_ids"] = sorted({
            c["chunk_id"] for c in selected
        })
        question_copy["relevance_status"] = "provisional_keyword_based"

    updated.append(question_copy)

with open(OUTPUT, "w") as f:
    json.dump(updated, f, indent=2, ensure_ascii=False)

print("Dataset generated successfully!")
print("Total questions:", len(updated))
print("Mapped questions:", len(updated) - len(missing))
print("Unmapped questions:", len(missing))
print("Unmapped IDs:", missing)
print("Output:", OUTPUT)
