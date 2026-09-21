import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import CrossEncoder

project_dir = Path(__file__).parent
results_dir = project_dir / "results"

# Read the saved hybrid candidates.
hybrid_data = json.loads(
    (results_dir / "hybrid_train.json").read_text(encoding="utf-8")
)

# Load document texts and training relevance labels.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

# Use the same reranker and settings as the earlier experiment.
model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
reranker = CrossEncoder(model_name, max_length=512)

outcomes = {
    "success_both": 0,
    "recovered": 0,
    "regressed": 0,
    "missed_both": 0,
}
query_results = {}

for number, (qid, query_text) in enumerate(queries.items(), start=1):
    # Get this query's hybrid top 100.
    candidate_ids = [
        item["doc_id"]
        for item in hybrid_data["queries"][qid]["top100"]
    ]

    # Pair the query with each candidate's title and abstract.
    pairs = [
        (
            query_text,
            f"{corpus[doc_id]['title']}\n{corpus[doc_id]['text']}",
        )
        for doc_id in candidate_ids
    ]

    # Score and reorder the candidates.
    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked_results = sorted(
        zip(candidate_ids, scores),
        key=lambda item: (-float(item[1]), item[0]),
    )

    final_ids = [doc_id for doc_id, _ in ranked_results[:10]]

    # Evaluate before and after reranking.
    relevant_ids = {
        doc_id
        for doc_id, label in qrels[qid].items()
        if label > 0
    }

    before_hit = bool(relevant_ids.intersection(candidate_ids[:10]))
    after_hit = bool(relevant_ids.intersection(final_ids))

    if before_hit and after_hit:
        outcome = "success_both"
    elif after_hit:
        outcome = "recovered"
    elif before_hit:
        outcome = "regressed"
    else:
        outcome = "missed_both"

    outcomes[outcome] += 1

    # Save individual rankings so we can inspect examples later.
    query_results[qid] = {
        "before_hit_at_10": before_hit,
        "after_hit_at_10": after_hit,
        "outcome": outcome,
        "top100": [
            {"doc_id": doc_id, "score": float(score)}
            for doc_id, score in ranked_results
        ],
    }

    if number % 50 == 0 or number == len(queries):
        print(f"Evaluated {number}/{len(queries)} queries", flush=True)

# Calculate overall Hit@10.
before_count = outcomes["success_both"] + outcomes["regressed"]
after_count = outcomes["success_both"] + outcomes["recovered"]

summary = {
    "dataset": "scifact",
    "split": "train",
    "method": "hybrid_rrf_then_cross_encoder",
    "reranker": model_name,
    "max_length": 512,
    "candidate_count": 100,
    "query_count": len(queries),
    "before_hit_at_10": before_count / len(queries),
    "after_hit_at_10": after_count / len(queries),
    "outcomes": outcomes,
}

# Save summary and per-query results.
output_path = results_dir / "hybrid_rerank_train.json"
output_path.write_text(
    json.dumps(
        {"summary": summary, "queries": query_results},
        indent=2,
    ),
    encoding="utf-8",
)

print("\nHybrid reranking results:")
print(f"Before reranking: {before_count}/{len(queries)}")
print(f"After reranking: {after_count}/{len(queries)}")
print(f"Before Hit@10: {summary['before_hit_at_10']:.2%}")
print(f"After Hit@10: {summary['after_hit_at_10']:.2%}")
print("Recovered:", outcomes["recovered"])
print("Regressed:", outcomes["regressed"])
print("Missed by both:", outcomes["missed_both"])
print("Saved results to:", output_path)