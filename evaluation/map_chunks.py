
import json
import re
from pathlib import Path
from collections import Counter

QUESTIONS_FILE = Path("evaluation/questions.json")
CORPUS_FILE = Path("evaluation/corpus_snapshot.json")
OUTPUT_FILE = Path("evaluation/reports/chunk_candidates.json")

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "of",
    "to", "and", "or", "for", "does", "how", "why",
    "it", "that", "this", "with", "from", "by", "be",
    "can", "which", "as", "on", "their", "its"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def calculate_score(query, chunk_text):
    query_words = Counter(tokenize(query))
    chunk_words = Counter(tokenize(chunk_text))

    if not query_words:
        return 0

    matched = sum(
        min(count, chunk_words[word])
        for word, count in query_words.items()
    )

    return round(matched / sum(query_words.values()), 4)


with open(QUESTIONS_FILE) as f:
    questions = json.load(f)

with open(CORPUS_FILE) as f:
    chunks = json.load(f)

results = []

for question in questions:
    query = (
        question["question"] + " " +
        question["expected_answer"]
    )

    candidates = []

    for chunk in chunks:
        if chunk["document_id"] != question["document_id"]:
            continue

        if chunk["page_number"] not in question["relevant_pages"]:
            continue

        score = calculate_score(query, chunk["text"])

        candidates.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "score": score,
            "text": chunk["text"]
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    results.append({
        "question_id": question["id"],
        "question": question["question"],
        "relevant_pages": question["relevant_pages"],
        "candidates": candidates[:5]
    })


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Chunk mapping candidates generated!")
print("Questions processed:", len(results))
print("Output:", OUTPUT_FILE)

for result in results:
    print(
        result["question_id"],
        "| Candidate chunks:",
        len(result["candidates"])
    )

