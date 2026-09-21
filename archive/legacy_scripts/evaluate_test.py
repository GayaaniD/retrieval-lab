import json
import re
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

project_dir = Path(__file__).parent
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

# Keep the settings from our training experiments.
embedding_model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
reranker_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
candidate_count = 100
rrf_constant = 60
k_values = [10, 100]


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Load test queries and labels, searching the same document collection.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="test")

print(f"\nTest queries: {len(queries)}", flush=True)

doc_ids = list(corpus.keys())
doc_texts = [
    f"{corpus[doc_id]['title']}\n{corpus[doc_id]['text']}"
    for doc_id in doc_ids
]
doc_positions = {
    doc_id: index for index, doc_id in enumerate(doc_ids)
}
query_ids = list(queries.keys())

# Build the same vector and keyword search methods.
model = SentenceTransformer(embedding_model)

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

similarities = util.cos_sim(query_embeddings, doc_embeddings)
dense_top = similarities.topk(k=candidate_count, dim=1)
dense_indices = dense_top.indices.tolist()
dense_scores = dense_top.values.tolist()

bm25 = BM25Okapi([tokenize(text) for text in doc_texts])
reranker = CrossEncoder(reranker_model, max_length=512)

runs = {
    "Vector search": {},
    "BM25": {},
    "Hybrid RRF": {},
    "Hybrid + reranking": {},
}

for number, qid in enumerate(query_ids, start=1):
    row = number - 1

    # Get the vector top 100.
    dense_items = [
        (doc_ids[index], float(score))
        for index, score in zip(dense_indices[row], dense_scores[row])
    ]

    # Get the BM25 top 100 with the same tie-breaking rule.
    keyword_scores = bm25.get_scores(tokenize(queries[qid]))
    keyword_indices = sorted(
        range(len(doc_ids)),
        key=lambda index: (-float(keyword_scores[index]), doc_ids[index]),
    )[:candidate_count]

    keyword_items = [
        (doc_ids[index], float(keyword_scores[index]))
        for index in keyword_indices
    ]

    # Combine both rankings using RRF.
    fusion_scores = {}

    for ranked_list in (dense_items, keyword_items):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            fusion_scores[doc_id] = (
                fusion_scores.get(doc_id, 0.0)
                + 1.0 / (rrf_constant + rank)
            )

    hybrid_items = sorted(
        fusion_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:candidate_count]

    # Rerank only the hybrid top 100.
    pairs = [
        (queries[qid], doc_texts[doc_positions[doc_id]])
        for doc_id, _ in hybrid_items
    ]

    new_scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    reranked_items = sorted(
        [
            (doc_id, float(score))
            for (doc_id, _), score in zip(hybrid_items, new_scores)
        ],
        key=lambda item: (-item[1], item[0]),
    )

    # Store rankings without using the relevance labels.
    for method, items in (
        ("Vector search", dense_items),
        ("BM25", keyword_items),
        ("Hybrid RRF", hybrid_items),
        ("Hybrid + reranking", reranked_items),
    ):
        runs[method][qid] = {
            "top100": [
                {"doc_id": doc_id, "score": score}
                for doc_id, score in items
            ]
        }

    if number % 50 == 0 or number == len(query_ids):
        print(f"Processed {number}/{len(query_ids)} test queries", flush=True)

# Save rankings before calculating metrics.
filenames = {
    "Vector search": "dense_test.json",
    "BM25": "bm25_test.json",
    "Hybrid RRF": "hybrid_test.json",
    "Hybrid + reranking": "hybrid_rerank_test.json",
}

settings = {
    "embedding_model": embedding_model,
    "reranker_model": reranker_model,
    "reranker_max_length": 512,
    "candidate_count": candidate_count,
    "rrf_constant": rrf_constant,
    "tokenization": "lowercase regex words; no stemming or stopword removal",
}

for method, filename in filenames.items():
    (results_dir / filename).write_text(
        json.dumps(
            {
                "dataset": "scifact",
                "split": "test",
                "method": method,
                "settings": settings,
                "queries": runs[method],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

# Use BEIR for every metric.
evaluator = EvaluateRetrieval(k_values=k_values)
all_metrics = {}

for method, rankings in runs.items():
    # Preserve the saved order using unique descending scores.
    results = {
        qid: {
            item["doc_id"]: float(len(record["top100"]) - position)
            for position, item in enumerate(record["top100"])
        }
        for qid, record in rankings.items()
    }

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

    metrics = {
        **accuracy,
        **recall,
        **precision,
        **mrr,
        **ndcg,
        **map_scores,
    }
    all_metrics[method] = metrics

    print(f"\n{method} — TEST")
    for name, value in metrics.items():
        print(f"{name}: {value:.5f}")

# Save test metrics separately from training metrics.
output_path = results_dir / "beir_metrics_test.json"
output_path.write_text(
    json.dumps(
        {
            "dataset": "scifact",
            "split": "test",
            "query_count": len(queries),
            "settings": settings,
            "k_values": k_values,
            "ranking_policy": "preserve saved order using ordinal scores",
            "methods": all_metrics,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved test metrics to:", output_path)
