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

# Connect to the existing index.
pc = Pinecone(api_key=api_key)
description = pc.describe_index("retrievallab-scifact-llama")
index = pc.Index(host=description.host)

# Read the saved dataset.
corpus, _, _ = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

# Prepare the first 10 documents.
records = []

for doc_id in list(corpus)[:10]:
    document = corpus[doc_id]

    records.append({
        "_id": doc_id,
        "text": f"{document['title']}\n{document['text']}",
        "title": document["title"],
    })

# Pinecone generates embeddings and stores the records.
index.upsert_records(
    namespace="scifact",
    records=records,
)

print(f"Upload accepted: {len(records)} documents")
print("Namespace: scifact")