import hashlib
import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import CrossEncoder

# Set file paths.
project_dir = Path(__file__).parent
results_dir = project_dir / "results"

source_path = results_dir / "llama_pinecone_train.json"
output_path = results_dir / "llama_rerank_train.json"
metrics_path = results_dir / "beir_metrics_llama_rerank_train.json"

# Load saved retrieval results.
source_bytes = source_path.read_bytes()
source = json.loads(source_bytes)
original = source["queries"]

# Load document text and relevance labels.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

# Check that every query has 100 valid candidates.
for query_id in queries:
    documents = original[query_id]["top100"]
    ids = [item["doc_id"] for item in documents]

    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError(f"Query {query_id}: expected 100 unique candidates.")

    if not set(ids).issubset(corpus):
        raise ValueError(f"Query {query_id}: unknown document IDs.")

settings = {
    "dataset": "scifact",
    "split": "train",
    "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "max_length": 512,
    "candidate_count": 100,
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
}

# Resume only when settings and source results match.
if output_path.exists():
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    if saved["settings"] != settings:
        raise ValueError("Saved reranking results use different settings or input.")
else:
    saved = {"settings": settings, "queries": {}}

print("\nTraining queries:", len(queries))
print("Previously reranked:", len(saved["queries"]))

model = CrossEncoder(
    settings["reranker"],
    max_length=settings["max_length"],
)

for number, (query_id, query_text) in enumerate(queries.items(), start=1):
    if query_id not in saved["queries"]:
        candidates = original[query_id]["top100"]

        # Pair the query with each candidate's title and abstract.
        pairs = [
            (
                query_text,
                f"{corpus[item['doc_id']]['title']}\n"
                f"{corpus[item['doc_id']]['text']}",
            )
            for item in candidates
        ]

        scores = model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        # Rank by cross-encoder score; break ties by document ID.
        reranked = [
            {
                "doc_id": item["doc_id"],
                "score": float(score),
            }
            for item, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: (-item["score"], item["doc_id"]))

        saved["queries"][query_id] = {"top100": reranked}

        # Save progress after each query.
        temporary_path = output_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(saved, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    if number % 50 == 0 or number == len(queries):
        print(f"Processed {number}/{len(queries)} queries")


# Evaluate rankings with BEIR.
def evaluate(rankings):
    # Ordinal scores preserve the saved ranking order.
    results = {
        query_id: {
            item["doc_id"]: float(100 - position)
            for position, item in enumerate(rankings[query_id]["top100"])
        }
        for query_id in queries
    }

    k_values = [10, 100]
    evaluator = EvaluateRetrieval(k_values=k_values)

    ndcg, map_scores, recall, precision = evaluator.evaluate(
        qrels,
        results,
        k_values,
        ignore_identical_ids=False,
    )

    mrr = evaluator.evaluate_custom(
        qrels, results, k_values, metric="mrr"
    )
    accuracy = evaluator.evaluate_custom(
        qrels, results, k_values, metric="acc"
    )

    return {**accuracy, **recall, **precision, **mrr, **ndcg, **map_scores}


before = evaluate(original)
after = evaluate(saved["queries"])

# Count changes in top-10 success for individual queries.
changes = {
    "successful_both": 0,
    "recovered": 0,
    "regressed": 0,
    "missed_both": 0,
}

for query_id in queries:
    relevant_ids = {
        doc_id
        for doc_id, label in qrels[query_id].items()
        if label > 0
    }

    before_ids = {
        item["doc_id"] for item in original[query_id]["top100"][:10]
    }
    after_ids = {
        item["doc_id"] for item in saved["queries"][query_id]["top100"][:10]
    }

    before_hit = bool(relevant_ids & before_ids)
    after_hit = bool(relevant_ids & after_ids)

    if before_hit and after_hit:
        changes["successful_both"] += 1
    elif after_hit:
        changes["recovered"] += 1
    elif before_hit:
        changes["regressed"] += 1
    else:
        changes["missed_both"] += 1

print("\nLlama reranking — TRAIN")
print(f"{'Metric':<16} {'Before':>10} {'After':>10}")

for name in before:
    print(f"{name:<16} {before[name]:>10.5f} {after[name]:>10.5f}")

print("\nTop-10 success changes:")
for name, count in changes.items():
    print(f"{name}: {count}")

# Save metrics and query-level outcome counts.
metrics_path.write_text(
    json.dumps(
        {
            "settings": settings,
            "query_count": len(queries),
            "before": before,
            "after": after,
            "changes": changes,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved rankings to:", output_path)
print("Saved metrics to:", metrics_path)