import json
import os
import time
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from dotenv import load_dotenv
from pinecone import Pinecone

# Set paths and search settings.
project_dir = Path(__file__).parent
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

ranking_path = results_dir / "llama_pinecone_train.json"
metrics_path = results_dir / "beir_metrics_llama_train.json"

index_name = "retrievallab-scifact-llama"
namespace = "scifact"
top_k = 100
k_values = [10, 100]

# Connect to Pinecone.
load_dotenv(project_dir / ".env")
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("Add PINECONE_API_KEY to your .env file.")

pc = Pinecone(api_key=api_key)
description = pc.describe_index(index_name)
index = pc.Index(host=description.host)

# Load training queries and their answer key.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

settings = {
    "dataset": "scifact",
    "split": "train",
    "model": "llama-text-embed-v2",
    "index": index_name,
    "namespace": namespace,
    "top_k": top_k,
}

# Resume previously completed queries.
if ranking_path.exists():
    saved = json.loads(ranking_path.read_text(encoding="utf-8"))

    if saved["settings"] != settings:
        raise ValueError("Saved results use different settings.")
else:
    saved = {"settings": settings, "queries": {}}

print("\nTraining queries:", len(queries))
print("Previously saved:", len(saved["queries"]))

for number, (query_id, query_text) in enumerate(queries.items(), start=1):
    if query_id not in saved["queries"]:

        # Retry rate limits and temporary server errors.
        for attempt in range(6):
            try:
                response = index.search(
                    namespace=namespace,
                    query={
                        "inputs": {"text": query_text},
                        "top_k": top_k,
                    },
                    fields=["title"],
                )
                break
            except Exception as error:
                status = (
                    getattr(error, "status_code", None)
                    or getattr(error, "status", None)
                )

                if str(status) not in {"429", "500", "502", "503", "504"}:
                    raise

                if attempt == 5:
                    raise

                delay = 60 if str(status) == "429" else 2 ** (attempt + 2)
                print(f"Temporary error. Retrying in {delay} seconds...")
                time.sleep(delay)

        if hasattr(response, "to_dict"):
            response = response.to_dict()

        # Use the field names confirmed in your SDK.
        hits = response["result"]["hits"]
        ranked_documents = [
            {
                "doc_id": hit["id_"],
                "score": float(hit["score_"]),
            }
            for hit in hits
        ]

        # Check that retrieval returned a complete candidate list.
        returned_ids = [item["doc_id"] for item in ranked_documents]

        if len(returned_ids) != top_k or len(set(returned_ids)) != top_k:
            raise ValueError(f"Query {query_id}: expected 100 unique results.")

        if not set(returned_ids).issubset(corpus):
            raise ValueError(f"Query {query_id}: unexpected document IDs.")

        saved["queries"][query_id] = {"top100": ranked_documents}

        # Replace the checkpoint only after writing it completely.
        temporary_path = ranking_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(saved, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(ranking_path)

    if number % 50 == 0 or number == len(queries):
        print(f"Processed {number}/{len(queries)} queries")

# Preserve returned ranking order, as in our previous evaluation.
evaluation_results = {}

for query_id in queries:
    documents = saved["queries"][query_id]["top100"]
    evaluation_results[query_id] = {
        item["doc_id"]: float(len(documents) - position)
        for position, item in enumerate(documents)
    }

# Calculate every metric using BEIR.
evaluator = EvaluateRetrieval(k_values=k_values)

ndcg, map_scores, recall, precision = evaluator.evaluate(
    qrels,
    evaluation_results,
    k_values,
    ignore_identical_ids=False,
)

mrr = evaluator.evaluate_custom(
    qrels, evaluation_results, k_values, metric="mrr"
)

accuracy = evaluator.evaluate_custom(
    qrels, evaluation_results, k_values, metric="acc"
)

metrics = {
    **accuracy,
    **recall,
    **precision,
    **mrr,
    **ndcg,
    **map_scores,
}

print("\nLlama + Pinecone — TRAIN")

for name, value in metrics.items():
    print(f"{name}: {value:.5f}")

# Save the evaluation separately from search results.
metrics_path.write_text(
    json.dumps(
        {
            "settings": settings,
            "query_count": len(queries),
            "k_values": k_values,
            "ranking_policy": "Preserve returned order using ordinal scores",
            "metrics": metrics,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved rankings to:", ranking_path)
print("Saved metrics to:", metrics_path)