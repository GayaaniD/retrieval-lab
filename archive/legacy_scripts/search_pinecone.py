import os
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from dotenv import load_dotenv
from pinecone import Pinecone

# Load the API key.
project_dir = Path(__file__).parent
load_dotenv(project_dir / ".env")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Add PINECONE_API_KEY to your .env file.")

# Connect to the Llama index.
pc = Pinecone(api_key=api_key)
description = pc.describe_index("retrievallab-scifact-llama")
index = pc.Index(host=description.host)

# Load queries and relevance labels.
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

query_id = "4"
query_text = queries[query_id]

# Pinecone embeds the query and retrieves 100 candidates.
response = index.search(
    namespace="scifact",
    query={
        "inputs": {"text": query_text},
        "top_k": 100,
    },
    fields=["title"],
)

# Convert SDK response objects to a dictionary if needed.
if hasattr(response, "to_dict"):
    response = response.to_dict()

from pprint import pprint

# Inspect one result to see its actual field names.
hits = response["result"]["hits"]
print("\nResponse type:", type(response))
print("Number of hits:", len(hits))

if hits:
    print("First hit type:", type(hits[0]))
    print("\nFirst hit:")
    pprint(hits[0])
else:
    print("No results returned.")
# Convert SDK field names to the names used below.
hits = [
    {
        "_id": hit["id_"],
        "_score": hit["score_"],
        "fields": hit["fields"],
    }
    for hit in response["result"]["hits"]
]

# Use labels only to check the retrieved results.
relevant_ids = {
    doc_id
    for doc_id, label in qrels[query_id].items()
    if label > 0
}

print("\nQuery:", query_text)
print("\nTop 10 Llama results:")

for rank, hit in enumerate(hits[:10], start=1):
    doc_id = hit["_id"]
    label = (
        "LABELLED RELEVANT"
        if doc_id in relevant_ids
        else "Not labelled relevant"
    )

    print(f"\n{rank}. Document ID: {doc_id}")
    print(f"Score: {hit['_score']:.4f} | {label}")
    print("Title:", hit["fields"]["title"])

# Find each relevant document's position in the results.
ranks = {
    hit["_id"]: rank
    for rank, hit in enumerate(hits, start=1)
}

print("\nRelevant-document positions:")

for doc_id in sorted(relevant_ids):
    rank = ranks.get(doc_id)

    print("\nDocument:", doc_id)
    print("Llama rank:", rank if rank else "Outside returned top 100")
    print("Found in top 10:", rank is not None and rank <= 10)