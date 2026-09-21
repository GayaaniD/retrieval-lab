# The reference function below is copied from the uploaded evaluator.
import sys
import unittest
from types import ModuleType
from unittest.mock import patch
from retrievallab.core import evaluate

REFERENCE = 'def evaluate(rankings):\n    # Preserve saved ranking order for BEIR.\n    results = {\n        query_id: {\n            item["doc_id"]: float(100 - position)\n            for position, item in enumerate(rankings[query_id]["top100"])\n        }\n        for query_id in queries\n    }\n\n    k_values = [10, 100]\n    evaluator = EvaluateRetrieval(k_values=k_values)\n\n    ndcg, map_scores, recall, precision = evaluator.evaluate(\n        qrels,\n        results,\n        k_values,\n        ignore_identical_ids=False,\n    )\n\n    mrr = evaluator.evaluate_custom(\n        qrels, results, k_values, metric="mrr"\n    )\n    accuracy = evaluator.evaluate_custom(\n        qrels, results, k_values, metric="acc"\n    )\n\n    return {**accuracy, **recall, **precision, **mrr, **ndcg, **map_scores}'

class ContractTests(unittest.TestCase):
    def test_shared_evaluator_makes_same_beir_calls_as_original(self):
        calls = []
        class FakeBEIR:
            def __init__(self, **kwargs):
                calls.append(('init', kwargs))
            def evaluate(self, *args, **kwargs):
                calls.append(('evaluate', args, kwargs))
                return ({'NDCG@10': .5}, {'MAP@10': .5}, {'Recall@10': .5}, {'P@10': .1})
            def evaluate_custom(self, *args, **kwargs):
                calls.append(('custom', args, kwargs))
                return {kwargs['metric']: .5}
        rankings = {'q': {'top100': [{'doc_id': str(i), 'score': -1.0} for i in range(100)]}}
        qrels = {'q': {'q': 1, '3': 1}}
        scope = {'EvaluateRetrieval': FakeBEIR, 'queries': {'q': 'claim'}, 'qrels': qrels}
        exec(REFERENCE, scope)
        expected = scope['evaluate'](rankings)
        original_calls = list(calls)
        calls.clear()
        module = ModuleType('beir.retrieval.evaluation')
        module.EvaluateRetrieval = FakeBEIR
        with patch.dict(sys.modules, {'beir.retrieval.evaluation': module}):
            actual = evaluate(rankings, qrels)
        self.assertEqual(actual, expected)
        self.assertEqual(calls, original_calls)
