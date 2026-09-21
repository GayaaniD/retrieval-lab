import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader

project_dir = Path(__file__).parent
results_dir = project_dir / "results"


# Read a saved experiment.
def read_results(filename):
    return json.loads(
        (results_dir / filename).read_text(encoding="utf-8")
    )["queries"]


# Load rankings from each stage.
dense = read_results("dense_train.json")
bm25 = read_results("bm25_train.json")
hybrid = read_results("hybrid_train.json")
reranked = read_results("hybrid_rerank_train.json")

# Load query texts, documents, and relevance labels.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")


# Find a document's rank in a saved top-100 list.
def get_rank(results, query_id, document_id):
    for rank, item in enumerate(results[query_id]["top100"], start=1):
        if item["doc_id"] == document_id:
            return rank
    return "Outside saved top 100"


for qid, query_text in queries.items():
    relevant_ids = {
        doc_id
        for doc_id, label in qrels[qid].items()
        if label > 0
    }

    dense_top10 = {
        item["doc_id"]
        for item in dense[qid]["top100"][:10]
    }

    final_top10 = [
        item["doc_id"]
        for item in reranked[qid]["top100"][:10]
    ]

    # Select a query that failed initially but succeeded finally.
    if not relevant_ids.intersection(dense_top10):
        recovered_ids = [
            doc_id for doc_id in final_top10
            if doc_id in relevant_ids
        ]

        if recovered_ids:
            doc_id = recovered_ids[0]

            print("\nQuery ID:", qid)
            print("Query:", query_text)
            print("\nRecovered document:", doc_id)
            print("Title:", corpus[doc_id]["title"])
            print("Vector rank:", get_rank(dense, qid, doc_id))
            print("BM25 rank:", get_rank(bm25, qid, doc_id))
            print("Hybrid rank:", get_rank(hybrid, qid, doc_id))
            print("Final reranked rank:", get_rank(reranked, qid, doc_id))
            break
else:
    print("No matching recovery example found.")