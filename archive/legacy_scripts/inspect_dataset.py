from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# Get the folder containing this script and define a "datasets" subfolder.
data_dir = Path(__file__).parent / "datasets"

# Create the datasets folder if it does not already exist.
data_dir.mkdir(exist_ok=True)

# URL of the SciFact dataset ZIP file.
# Python automatically joins these two adjacent strings into one URL.
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

# Download and extract SciFact.
# Store the path to the extracted dataset folder in data_path.
data_path = util.download_and_unzip(url, str(data_dir))

# Load the document collection and the training queries with their labels.
# The test queries are reserved for final evaluation.
#
# corpus: document ID -> {"title": ..., "text": ...}
# queries: query ID -> query text
# qrels: query ID -> {document ID: relevance label}
corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="train")

# Display the number of documents, queries, and labelled queries.
# "\n" adds a blank line before the output.
print("\nDocuments:", len(corpus))
print("Queries:", len(queries))
print("Queries with relevance labels:", len(qrels))

# Get the first query ID in the queries dictionary.
# iter() creates an iterator over its keys; next() gets the first key.
# This selects the first query, not a random query.
query_id = next(iter(queries))

# Display the selected query's ID and text.
print("\nQuery ID:", query_id)
print("Query:", queries[query_id])

# Get the document IDs and relevance labels associated with this query.
# A query may have more than one labelled document.
for doc_id, relevance in qrels[query_id].items():

    # A positive relevance label identifies a relevant document.
    if relevance > 0:

        # Look up the document's title and text using its ID.
        document = corpus[doc_id]

        # Display the labelled relevant document.
        print("\nRelevant document ID:", doc_id)
        print("Title:", document["title"])
        print("Text:", document["text"])
