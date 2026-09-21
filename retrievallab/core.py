"""Shared ranking, persistence and benchmark evaluation helpers."""
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

EMBEDDER = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L6-v2'
K = 100

def text_of(doc):
    return f"{doc['title']}\n{doc['text']}"

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def load_dataset(root, split):
    from beir.datasets.data_loader import GenericDataLoader
    return GenericDataLoader(data_folder=str(root / 'datasets' / 'scifact')).load(split=split)

def validate(rankings, queries, corpus, complete=True):
    if set(rankings) - set(queries):
        raise ValueError('Ranking file contains unexpected query IDs for this split.')
    if complete and set(queries) - set(rankings):
        raise ValueError('Ranking file is incomplete for this split.')
    for qid, record in rankings.items():
        items = record['top100']
        ids = [item['doc_id'] for item in items]
        if len(ids) != K or len(set(ids)) != K:
            raise ValueError(f'{qid}: expected 100 unique candidates.')
        if not set(ids).issubset(corpus):
            raise ValueError(f'{qid}: unknown document IDs.')
        if not all(math.isfinite(float(item['score'])) for item in items):
            raise ValueError(f'{qid}: non-finite scores.')

def rankings_from(path, split, queries, corpus):
    data = read(path)
    # Older files place metadata in different locations.
    for metadata in (data, data.get('settings', {}), data.get('summary', {})):
        if metadata.get('split', split) != split:
            raise ValueError(f'{path}: split mismatch.')
    rankings = data['queries']
    validate(rankings, queries, corpus)
    return rankings

def sorted_items(ids, scores):
    items = [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in zip(ids, scores, strict=True)]
    return sorted(items, key=lambda item: (-item['score'], item['doc_id']))

def fuse(*lists):
    scores = defaultdict(float)
    for items in lists:
        for rank, item in enumerate(items, start=1):
            scores[item['doc_id']] += 1 / (60 + rank)
    return sorted_items(list(scores), list(scores.values()))[:K]

def ordinal(rankings):
    return {qid: {item['doc_id']: float(len(record['top100']) - pos)
                  for pos, item in enumerate(record['top100'])}
            for qid, record in rankings.items()}

def evaluate(rankings, qrels):
    from beir.retrieval.evaluation import EvaluateRetrieval
    evaluator = EvaluateRetrieval(k_values=[10, 100])
    results = ordinal(rankings)
    ndcg, maps, recall, precision = evaluator.evaluate(qrels, results, [10, 100], ignore_identical_ids=False)
    mrr = evaluator.evaluate_custom(qrels, results, [10, 100], metric='mrr')
    accuracy = evaluator.evaluate_custom(qrels, results, [10, 100], metric='acc')
    return {**accuracy, **recall, **precision, **mrr, **ndcg, **maps}

def changes(before, after, qrels):
    counts = dict(successful_both=0, recovered=0, regressed=0, missed_both=0)
    for qid, labels in qrels.items():
        relevant = {doc_id for doc_id, label in labels.items() if label > 0}
        b = bool(relevant & {d['doc_id'] for d in before[qid]['top100'][:10]})
        a = bool(relevant & {d['doc_id'] for d in after[qid]['top100'][:10]})
        key = 'successful_both' if a and b else 'recovered' if a else 'regressed' if b else 'missed_both'
        counts[key] += 1
    return counts
