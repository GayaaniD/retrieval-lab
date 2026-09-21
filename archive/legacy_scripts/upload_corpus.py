import os
import time
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
namespace = "scifact"

# Read the complete document collection.
corpus, _, _ = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="train")

doc_ids = list(corpus)
batch_size = 32

# Upload documents in small batches.
# for start in range(0, len(doc_ids), batch_size):
# Resume after the 4,608 documents already accepted.
for start in range(4608, len(doc_ids), batch_size):
    batch_ids = doc_ids[start:start + batch_size]

    records = [
        {
            "_id": doc_id,
            "text": f"{corpus[doc_id]['title']}\n{corpus[doc_id]['text']}",
            "title": corpus[doc_id]["title"],
        }
        for doc_id in batch_ids
    ]

    # Retry temporary server errors and rate limits.
    for attempt in range(6):
        try:
            index.upsert_records(
                namespace=namespace,
                records=records,
            )
            break
        # except Exception as error:
        #     status = getattr(error, "status", None)

        #     if status not in (429, 500, 502, 503, 504) or attempt == 5:
        #         raise

        #     delay = min(2 ** (attempt + 2), 60)
        #     print(f"Temporary error. Retrying in {delay} seconds...")
        #     time.sleep(delay)
        except Exception as error:
            # SDK versions may expose different status attributes.
            status = (
                getattr(error, "status_code", None)
                or getattr(error, "status", None)
            )

            if str(status) not in {"429", "500", "502", "503", "504"}:
                raise

            if attempt == 5:
                raise

            # Allow the per-minute quota time to recover.
            delay = 60 if str(status) == "429" else min(2 ** (attempt + 2), 60)
            print(f"Temporary error. Retrying in {delay} seconds...")
            time.sleep(delay)

    uploaded = start + len(batch_ids)
    print(f"Upload accepted: {uploaded}/{len(doc_ids)}")

print("\nAll batches accepted.")
print("Refresh Pinecone to check the record count.")