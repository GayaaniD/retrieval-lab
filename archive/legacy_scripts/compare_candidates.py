import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, util

project_dir = Path(__file__).parent
results_dir = project_dir / "results"

# Read our saved BM25 rankings.
bm25_results = json.loads(
    (results_dir / "bm25_train.json").read_text(encoding="utf-8")
)

# Load the same training dataset.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

doc_ids = list(corpus.keys())
doc_texts = [
    f"{corpus[doc_id]['title']}\n{corpus[doc_id]['text']}"
    for doc_id in doc_ids
]
query_ids = list(queries.keys())

# Use the same embedding model as before.
model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
model = SentenceTransformer(model_name)

doc_embeddings = model.encode(
    doc_texts,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True,
)

query_embeddings = model.encode(
    [queries[qid] for qid in query_ids],
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True,
)

# Retrieve the top 100 vector-search candidates per query.
scores = util.cos_sim(query_embeddings, doc_embeddings)
top100 = scores.topk(k=100, dim=1)
indices_by_query = top100.indices.tolist()
scores_by_query = top100.values.tolist()

counts = {
    "both": 0,
    "vector_only": 0,
    "bm25_only": 0,
    "neither": 0,
}
dense_results = {}

for qid, indices, values in zip(
    query_ids, indices_by_query, scores_by_query
):
    # Save vector rankings for reuse.
    dense_results[qid] = {
        "top100": [
            {"doc_id": doc_ids[index], "score": float(score)}
            for index, score in zip(indices, values)
        ]
    }

    vector_ids = {doc_ids[index] for index in indices}
    keyword_ids = {
        item["doc_id"]
        for item in bm25_results["queries"][qid]["top100"]
    }

    # Evaluate each candidate pool using relevance labels.
    relevant_ids = {
        doc_id
        for doc_id, label in qrels[qid].items()
        if label > 0
    }

    vector_hit = bool(relevant_ids & vector_ids)
    bm25_hit = bool(relevant_ids & keyword_ids)

    if vector_hit and bm25_hit:
        counts["both"] += 1
    elif vector_hit:
        counts["vector_only"] += 1
    elif bm25_hit:
        counts["bm25_only"] += 1
    else:
        counts["neither"] += 1

# Save vector rankings and the comparison.
output = {
    "dataset": "scifact",
    "split": "train",
    "model": model_name,
    "candidate_count": 100,
    "comparison": counts,
    "queries": dense_results,
}
(results_dir / "dense_train.json").write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

combined_hits = len(query_ids) - counts["neither"]

print("\nCandidate-pool comparison:")
print("Evidence found by both:", counts["both"])
print("Evidence found only by vector search:", counts["vector_only"])
print("Evidence found only by BM25:", counts["bm25_only"])
print("Evidence missed by both:", counts["neither"])
print(f"Combined candidate Hit rate: {combined_hits / len(query_ids):.2%}")
print("Saved vector rankings to results/dense_train.json")