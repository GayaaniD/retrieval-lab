import re
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Convert text into lowercase words and numbers.
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Read the saved dataset files into memory.
data_path = Path(__file__).parent / "datasets" / "scifact"
corpus, queries, qrels = GenericDataLoader(
    data_folder=str(data_path)
).load(split="train")

# Keep document IDs aligned with their texts and embeddings.
doc_ids = list(corpus.keys())

# Combine each document's title and abstract.
doc_texts = [
    f"{corpus[doc_id]['title']}\n{corpus[doc_id]['text']}"
    for doc_id in doc_ids
]

# Map document IDs to their positions in the lists.
doc_positions = {
    doc_id: index
    for index, doc_id in enumerate(doc_ids)
}

# Select the query to search.
query_id = "0"
query_text = queries[query_id]

# Collect labelled relevant documents for evaluation only.
relevant_ids = {
    doc_id
    for doc_id, label in qrels[query_id].items()
    if label > 0
}

# Load the embedding model; download it if not cached.
model = SentenceTransformer(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

# Create document embeddings in batches of 32.
doc_embeddings = model.encode(
    doc_texts,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True,
)

# Create an embedding for the query.
query_embedding = model.encode(
    query_text,
    convert_to_tensor=True,
)

# Compare the query with every document.
scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# Get the 10 highest scores and their document positions.
top_results = scores.topk(k=10)

print("\nQuery:", query_text)
print("\nTop 10 vector-search results:")

# Collect retrieved document IDs for evaluation.
retrieved_ids = set()

for rank, (score, index) in enumerate(
    zip(top_results.values.tolist(), top_results.indices.tolist()),
    start=1,
):
    # Convert the position back to the document ID.
    doc_id = doc_ids[index]
    retrieved_ids.add(doc_id)

    # Check relevance after retrieval.
    is_relevant = doc_id in relevant_ids
    label = "LABELLED RELEVANT" if is_relevant else "Not labelled relevant"

    print(f"\n{rank}. Document ID: {doc_id}")
    print(f"Similarity: {score:.4f} | {label}")
    print("Title:", corpus[doc_id]["title"])

# Show which relevant documents were found or missed.
print("\nRelevant IDs:", sorted(relevant_ids))
print("Found in top 10:", sorted(relevant_ids & retrieved_ids))
print("Missing from top 10:", sorted(relevant_ids - retrieved_ids))

# Sort all documents by vector similarity.
ranked_indices = scores.argsort(descending=True).tolist()

# Map document IDs to their vector-search ranks.
doc_ranks = {
    doc_ids[index]: rank
    for rank, index in enumerate(ranked_indices, start=1)
}

# Show the relevant documents' ranks across the full corpus.
for doc_id in sorted(relevant_ids):
    position = doc_positions[doc_id]

    print("\nRelevant document:", doc_id)
    print("Rank among all documents:", doc_ranks[doc_id])
    print(f"Similarity: {scores[position].item():.4f}")

# Build a BM25 keyword-search index.
tokenized_docs = [tokenize(text) for text in doc_texts]
bm25 = BM25Okapi(tokenized_docs)

# Calculate keyword-search scores for the same query.
bm25_scores = bm25.get_scores(tokenize(query_text))

# Sort by score; break ties using document IDs.
bm25_indices = sorted(
    range(len(doc_ids)),
    key=lambda index: (-bm25_scores[index], doc_ids[index]),
)

# Map document IDs to their BM25 ranks.
bm25_ranks = {
    doc_ids[index]: rank
    for rank, index in enumerate(bm25_indices, start=1)
}

# Compare vector-search and keyword-search ranks.
for doc_id in sorted(relevant_ids):
    print("\nRelevant document:", doc_id)
    print("Vector-search rank:", doc_ranks[doc_id])
    print("BM25 rank:", bm25_ranks[doc_id])
    print("In BM25 top 100:", bm25_ranks[doc_id] <= 100)

# Inspect keyword overlap for the relevant documents.
query_words = set(tokenize(query_text))
print("\nQuery words:", sorted(query_words))

for doc_id in sorted(relevant_ids):
    position = doc_positions[doc_id]
    document_text = doc_texts[position]
    document_words = set(tokenize(document_text))
    bm25_score = bm25_scores[position]

    print("\nRelevant document:", doc_id)
    print("Document text:", document_text)
    print("Matching words:", sorted(query_words & document_words))
    print(f"BM25 score: {bm25_score:.4f}")

    # Count documents tied at this BM25 score.
    print(
        "Documents with the same score:",
        int((bm25_scores == bm25_score).sum()),
    )

# Embed all training queries using the same model.
query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]

query_embeddings = model.encode(
    query_texts,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True,
)

# Compare every query against the existing document embeddings.
all_scores = util.cos_sim(query_embeddings, doc_embeddings)

# Get the top 100 document positions for each query.
top100_indices = all_scores.topk(k=100, dim=1).indices.tolist()

missed_top10 = 0
potential_cases = []

for qid, indices in zip(query_ids, top100_indices):
    # Get the labelled relevant documents for this query.
    expected_ids = {
        doc_id
        for doc_id, label in qrels[qid].items()
        if label > 0
    }

    ranked_ids = [doc_ids[index] for index in indices]

    # Find queries with no labelled relevant evidence in the top 10.
    if not expected_ids.intersection(ranked_ids[:10]):
        missed_top10 += 1

        # Check whether relevant evidence appears at ranks 11–100.
        if expected_ids.intersection(ranked_ids):
            potential_cases.append((qid, ranked_ids, expected_ids))

print("\nTraining queries checked:", len(query_ids))
print("Queries with no relevant evidence in top 10:", missed_top10)
print("Of those, evidence present in top 100:", len(potential_cases))

# Show the first qualifying example.
if potential_cases:
    qid, ranked_ids, expected_ids = potential_cases[0]

    print("\nExample query ID:", qid)
    print("Query:", queries[qid])

    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in expected_ids:
            print("Relevant document:", doc_id)
            print("Current rank:", rank)
            print("Title:", corpus[doc_id]["title"])
else:
    print("No qualifying examples found.")


# Select query 4 from the queries we already searched.
rerank_query_id = "4"
query_position = query_ids.index(rerank_query_id)
rerank_query_text = queries[rerank_query_id]

# Get its top 100 vector-search candidates.
candidate_indices = top100_indices[query_position]
candidate_ids = [doc_ids[index] for index in candidate_indices]

# Pair the query with each candidate's title and abstract.
pairs = [
    (rerank_query_text, doc_texts[index])
    for index in candidate_indices
]

# Load the reranker; download it on the first run.
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    max_length=512,
)

# Score all 100 query-document pairs.
rerank_scores = reranker.predict(
    pairs,
    batch_size=16,
    show_progress_bar=True,
)

# Sort candidates by their new scores.
reranked_results = sorted(
    zip(candidate_ids, rerank_scores),
    key=lambda item: (-float(item[1]), item[0]),
)

# Use relevance labels only to evaluate the ranking.
expected_ids = {
    doc_id
    for doc_id, label in qrels[rerank_query_id].items()
    if label > 0
}

print("\nReranking query:", rerank_query_text)
print("\nTop 10 after reranking:")

for rank, (doc_id, score) in enumerate(reranked_results[:10], start=1):
    label = (
        "LABELLED RELEVANT"
        if doc_id in expected_ids
        else "Not labelled relevant"
    )

    print(f"\n{rank}. Document ID: {doc_id}")
    print(f"Reranker score: {float(score):.4f} | {label}")
    print("Title:", corpus[doc_id]["title"])

# Compare ranks for relevant documents in the candidate pool.
print("\nRelevant-document rank changes:")

for new_rank, (doc_id, _) in enumerate(reranked_results, start=1):
    if doc_id in expected_ids:
        old_rank = candidate_ids.index(doc_id) + 1

        print("\nDocument:", doc_id)
        print("Before reranking:", old_rank)
        print("After reranking:", new_rank)
        print("Now in top 10:", new_rank <= 10)

# Count query outcomes before and after reranking.
outcomes = {
    "success_both": 0,
    "recovered": 0,
    "regressed": 0,
    "missed_both": 0,
}

for number, (qid, indices) in enumerate(
    zip(query_ids, top100_indices),
    start=1,
):
    candidate_ids = [doc_ids[index] for index in indices]

    # Labels are used only to evaluate the results.
    expected_ids = {
        doc_id
        for doc_id, label in qrels[qid].items()
        if label > 0
    }

    # Did vector search find any relevant evidence in its top 10?
    before_hit = bool(expected_ids.intersection(candidate_ids[:10]))

    # Rerank this query's 100 candidates.
    pairs = [
        (queries[qid], doc_texts[index])
        for index in indices
    ]

    new_scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked_candidates = sorted(
        zip(candidate_ids, new_scores),
        key=lambda item: (-float(item[1]), item[0]),
    )

    final_ids = [doc_id for doc_id, _ in ranked_candidates[:10]]

    # Did reranking find any relevant evidence in its top 10?
    after_hit = bool(expected_ids.intersection(final_ids))

    if before_hit and after_hit:
        outcomes["success_both"] += 1
    elif not before_hit and after_hit:
        outcomes["recovered"] += 1
    elif before_hit and not after_hit:
        outcomes["regressed"] += 1
    else:
        outcomes["missed_both"] += 1

    # Print progress periodically.
    if number % 50 == 0 or number == len(query_ids):
        print(f"Evaluated {number}/{len(query_ids)} queries", flush=True)

# Hit@10: percentage of queries with at least one relevant result.
total = len(query_ids)
before_count = outcomes["success_both"] + outcomes["regressed"]
after_count = outcomes["success_both"] + outcomes["recovered"]

print("\nOverall training results:")
print(f"Vector-search Hit@10: {before_count / total:.2%}")
print(f"After-reranking Hit@10: {after_count / total:.2%}")

print("\nSuccessful with both:", outcomes["success_both"])
print("Recovered by reranking:", outcomes["recovered"])
print("Made worse by reranking:", outcomes["regressed"])
print("Missed by both:", outcomes["missed_both"])
