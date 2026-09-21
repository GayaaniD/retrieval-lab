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

# Connect to the index.
pc = Pinecone(api_key=api_key)
description = pc.describe_index("retrievallab-scifact-llama")
index = pc.Index(host=description.host)

# Get the expected document IDs.
corpus, _, _ = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

expected_ids = set(corpus)

# Read all stored IDs, page by page.
stored_ids = set()

# Extract document IDs from each returned page.
for page in index.list(namespace="scifact"):
    for item in page:
        doc_id = item if isinstance(item, str) else item.id
        stored_ids.add(doc_id)

# Compare the two collections.
missing_ids = expected_ids - stored_ids
unexpected_ids = stored_ids - expected_ids

print("\nExpected documents:", len(expected_ids))
print("Stored documents:", len(stored_ids))
print("Missing documents:", len(missing_ids))
print("Unexpected documents:", len(unexpected_ids))

if not missing_ids and not unexpected_ids:
    print("PASS: All expected document IDs are present.")
else:
    print("Missing IDs (first 20):", sorted(missing_ids)[:20])
    print("Unexpected IDs (first 20):", sorted(unexpected_ids)[:20])
