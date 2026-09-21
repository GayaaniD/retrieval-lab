import json
from collections import defaultdict
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Set paths.
project_dir = Path(__file__).parent
results_dir = project_dir / "results"

llama_path = results_dir / "llama_pinecone_train.json"
bm25_path = results_dir / "bm25_train.json"

# Load saved rankings.
llama = json.loads(llama_path.read_text(encoding="utf-8"))["queries"]
bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))["queries"]

# Load the answer key for evaluation.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

rrf_constant = 60
hybrid = {}

for query_id in queries:
    fusion_scores = defaultdict(float)

    for method in (llama, bm25):
        candidates = method[query_id]["top100"]
        ids = [item["doc_id"] for item in candidates]

        # Check that both methods have complete, valid results.
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(f"Query {query_id}: expected 100 unique candidates.")

        if not set(ids).issubset(corpus):
            raise ValueError(f"Query {query_id}: unknown document IDs.")

        # Each method contributes points based on rank.
        for rank, doc_id in enumerate(ids, start=1):
            fusion_scores[doc_id] += 1 / (rrf_constant + rank)

    # Sort combined candidates and keep the best 100.
    ranked = sorted(
        fusion_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:100]

    hybrid[query_id] = {
        "top100": [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in ranked
        ]
    }


def evaluate(rankings):
    # Preserve saved ranking order for BEIR.
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


# Compare Llama alone with the new hybrid ranking.
before = evaluate(llama)
after = evaluate(hybrid)

print("\nLlama vs Llama + BM25 — TRAIN")
print(f"{'Metric':<16} {'Llama':>10} {'Hybrid':>10}")

for name in before:
    print(f"{name:<16} {before[name]:>10.5f} {after[name]:>10.5f}")

# Save rankings for later reranking experiments.
ranking_path = results_dir / "llama_hybrid_train.json"
ranking_path.write_text(
    json.dumps(
        {
            "dataset": "scifact",
            "split": "train",
            "rrf_constant": rrf_constant,
            "candidate_count_per_method": 100,
            "queries": hybrid,
        },
        indent=2,
    ),
    encoding="utf-8",
)

# Save standard BEIR metrics.
metrics_path = results_dir / "beir_metrics_llama_hybrid_train.json"
metrics_path.write_text(
    json.dumps(
        {
            "dataset": "scifact",
            "split": "train",
            "query_count": len(queries),
            "llama": before,
            "llama_hybrid": after,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved rankings to:", ranking_path)
print("Saved metrics to:", metrics_path)
