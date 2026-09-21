import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

# Load the API key from this project's .env file.
load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("PINECONE_API_KEY")

# Stop if the key is missing.
if not api_key:
    raise ValueError("Add PINECONE_API_KEY to your .env file.")

# Create the Pinecone client.
pc = Pinecone(api_key=api_key)

# Make a request to verify the connection.
index_names = pc.list_indexes().names()

print("Pinecone connection successful!")
print("Indexes found:", len(index_names))
