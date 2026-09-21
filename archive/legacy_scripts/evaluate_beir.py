import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

project_dir = Path(__file__).parent
results_dir = project_dir / "results"

# Load the relevance labels for our training queries.
_, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

# Evaluate the four methods with saved per-query rankings.
experiments = {
    "Vector search": "dense_train.json",
    "BM25": "bm25_train.json",
    "Hybrid RRF": "hybrid_train.json",
    "Hybrid + reranking": "hybrid_rerank_train.json",
}

k_values = [10, 100]
evaluator = EvaluateRetrieval(k_values=k_values)
all_metrics = {}

for method, filename in experiments.items():
    saved_data = json.loads(
        (results_dir / filename).read_text(encoding="utf-8")
    )
    rankings = saved_data["queries"]

    # Check that every query has a saved ranking.
    missing_queries = set(queries) - set(rankings)
    if missing_queries:
        raise ValueError(
            f"{method}: missing rankings for {len(missing_queries)} queries"
        )

    # Convert saved rankings to BEIR's query -> document -> score format.
    results = {}

    for qid in queries:
        items = rankings[qid]["top100"]
        doc_ids = [item["doc_id"] for item in items]

        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"{method}: duplicate documents for query {qid}")

        if len(doc_ids) < max(k_values):
            raise ValueError(f"{method}: fewer than 100 results for query {qid}")

        # Unique descending scores preserve the exact saved order.
        # This avoids different tie-breaking between metric implementations.
        results[qid] = {
            doc_id: float(len(doc_ids) - position)
            for position, doc_id in enumerate(doc_ids)
        }

    # BEIR calculates NDCG, MAP, Recall, and Precision.
    ndcg, map_scores, recall, precision = evaluator.evaluate(
        qrels,
        results,
        k_values,
        ignore_identical_ids=False,
    )

    # BEIR calculates MRR.
    mrr = evaluator.evaluate_custom(
        qrels,
        results,
        k_values,
        metric="mrr",
    )

    # BEIR's top-k accuracy is our Hit@k metric.
    accuracy = evaluator.evaluate_custom(
        qrels,
        results,
        k_values,
        metric="acc",
    )

    # Collect the metric dictionaries returned by BEIR.
    metrics = {
        **accuracy,
        **recall,
        **precision,
        **mrr,
        **ndcg,
        **map_scores,
    }
    all_metrics[method] = metrics

    print(f"\n{method}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.5f}")

# Save all methods' metrics in one file.
output_path = results_dir / "beir_metrics_train.json"
output_path.write_text(
    json.dumps(
        {
            "dataset": "scifact",
            "split": "train",
            "query_count": len(queries),
            "k_values": k_values,
            "ranking_policy": "preserve saved order using ordinal scores",
            "methods": all_metrics,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved results to:", output_path)
