import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader

project_dir = Path(__file__).parent

# Check the existing BM25 test results.
bm25_path = project_dir / "results" / "bm25_test.json"

if not bm25_path.exists():
    raise FileNotFoundError(
        "Missing results/bm25_test.json. "
        "This should come from the earlier test evaluation."
    )

_, queries, _ = GenericDataLoader(
    data_folder=str(project_dir / "datasets" / "scifact")
).load(split="test")

bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))["queries"]

for query_id in queries:
    documents = bm25[query_id]["top100"]
    ids = [item["doc_id"] for item in documents]

    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError(f"Invalid BM25 candidates for query {query_id}.")

print(f"BM25 results available for all {len(queries)} test queries.")

# Create test versions of our working training scripts.
script_pairs = {
    "evaluate_pinecone.py": "evaluate_pinecone_test.py",
    "rerank_llama.py": "rerank_llama_test.py",
    "evaluate_llama_hybrid.py": "evaluate_llama_hybrid_test.py",
    "rerank_llama_hybrid.py": "rerank_llama_hybrid_test.py",
}

for source_name, target_name in script_pairs.items():
    source_path = project_dir / source_name
    code = source_path.read_text(encoding="utf-8")

    # Change the split, result filenames, and display labels.
    code = code.replace('"train"', '"test"')
    code = code.replace("'train'", "'test'")
    code = code.replace("_train.json", "_test.json")
    code = code.replace("Training queries:", "Test queries:")
    code = code.replace("— TRAIN", "— TEST")
    code = code.replace("training queries", "test queries")

    target_path = project_dir / target_name
    target_path.write_text(code, encoding="utf-8")

    print("Created:", target_name)
