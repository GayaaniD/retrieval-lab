import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from retrievallab.core import fuse, sorted_items, validate, ordinal, changes, rankings_from, save
from retrievallab.pinecone_io import search, stored_ids, retry


def record(ids):
    return {'top100': [{'doc_id': str(i), 'score': 0.0} for i in ids]}


class RegressionTests(unittest.TestCase):
    def test_fusion_matches_original_formula_with_duplicates_and_ties(self):
        left = record(range(100))['top100']
        right = record(range(50, 150))['top100']
        expected = {}
        for candidates in (left, right):
            for rank, item in enumerate(candidates, 1):
                d = item['doc_id']
                expected[d] = expected.get(d, 0.0) + 1.0 / (60 + rank)
        expected = sorted(expected.items(), key=lambda item: (-item[1], item[0]))[:100]
        actual = fuse(left, right)
        self.assertEqual(actual, [{'doc_id': d, 'score': s} for d, s in expected])
        self.assertEqual(len({x['doc_id'] for x in actual}), 100)

    def test_ordinal_preserves_saved_order_when_scores_tie_or_are_negative(self):
        ranks = {'q': {'top100': [{'doc_id': 'z', 'score': -1}, {'doc_id': 'a', 'score': -1}]}}
        self.assertEqual(ordinal(ranks), {'q': {'z': 2.0, 'a': 1.0}})
        self.assertEqual([x['doc_id'] for x in sorted_items(['z', 'a'], [-1, -1])], ['a', 'z'])

    def test_validation_rejects_missing_duplicate_unknown_nonfinite(self):
        corpus = {str(i): {} for i in range(101)}
        validate({'q': record(range(100))}, {'q': 'query'}, corpus)
        for ranks in ({}, {'q': record([0] * 100)}, {'q': record(range(1, 102))}):
            with self.assertRaises(ValueError):
                validate(ranks, {'q': 'query'}, corpus)
        bad = record(range(100)); bad['top100'][0]['score'] = float('nan')
        with self.assertRaises(ValueError):
            validate({'q': bad}, {'q': 'query'}, corpus)

    def test_old_metadata_and_split_isolation(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            save(path, {'summary': {'split': 'test'}, 'queries': {'q': record(range(100))}})
            corpus = {str(i): {} for i in range(100)}
            self.assertEqual(len(rankings_from(path, 'test', {'q': ''}, corpus)), 1)
            with self.assertRaises(ValueError):
                rankings_from(path, 'train', {'q': ''}, corpus)

    def test_outcome_counts(self):
        before = {q: record(range(100)) for q in ['both', 'recover', 'regress', 'neither']}
        after = {q: record(range(100)) for q in before}
        after['recover'] = record([20] + list(range(20)) + list(range(21, 100)))
        after['regress'] = record(list(range(1, 100)) + [0])
        labels = {'both': {'0': 1}, 'recover': {'20': 1}, 'regress': {'0': 1}, 'neither': {'99': 1}}
        self.assertEqual(changes(before, after, labels), dict(successful_both=1, recovered=1, regressed=1, missed_both=1))

    def test_pinecone_response_formats(self):
        for id_key, score_key in [('id_', 'score_'), ('_id', '_score'), ('id', 'score')]:
            response = {'result': {'hits': [{id_key: '42', score_key: .25}]}}
            client = SimpleNamespace(search=lambda **kwargs: response)
            self.assertEqual(search(client, 'scifact', 'claim'), [{'doc_id': '42', 'score': .25}])
        client = SimpleNamespace(list=lambda **kwargs: [['1', SimpleNamespace(id='2')], [{'id': '3'}]])
        self.assertEqual(stored_ids(client, 'scifact'), {'1', '2', '3'})

    def test_retry_handles_both_status_attributes_and_is_bounded(self):
        class Failure(Exception): pass
        for attribute in ['status', 'status_code']:
            error = Failure(); setattr(error, attribute, 429)
            attempts, waits = [], []
            def operation():
                attempts.append(1)
                raise error
            with self.assertRaises(Failure):
                retry(operation, sleep=waits.append)
            self.assertEqual(len(attempts), 6)
            self.assertEqual(waits, [60] * 5)
        error = Failure(); error.status_code = 401
        with self.assertRaises(Failure):
            retry(lambda: (_ for _ in ()).throw(error), sleep=lambda _: self.fail('must not retry 401'))


if __name__ == '__main__':
    unittest.main()
