import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from retrievallab.core import save, read, fuse
from retrievallab.runner import run


class RunnerTests(unittest.TestCase):
    def test_hybrid_resume_and_source_invalidation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = {str(i): {'title': str(i), 'text': 'abstract'} for i in range(150)}
            queries = {'q': 'claim'}
            qrels = {'q': {'1': 1}}
            def ranking(ids):
                return {'queries': {'q': {'top100': [{'doc_id': str(i), 'score': 0.0} for i in ids]}}, 'split': 'train'}
            left, right = ranking(range(100)), ranking(range(50, 150))
            save(root/'left.json', left); save(root/'right.json', right)
            args = SimpleNamespace(method='hybrid', split='train', source=root/'left.json', other=root/'right.json',
                                   project=root, output=root/'output.json')
            with patch('retrievallab.runner.evaluate', return_value={'Accuracy@10': 1.0}):
                run(args, corpus, queries, qrels)
                result = read(args.output)
                self.assertEqual(result['queries']['q']['top100'], fuse(left['queries']['q']['top100'],right['queries']['q']['top100']))
                before = args.output.read_bytes()
                with patch('retrievallab.runner.fuse', side_effect=AssertionError('must not repeat completed queries')):
                    run(args, corpus, queries, qrels)
                self.assertEqual(before, args.output.read_bytes())
                left['queries']['q']['top100'].reverse()
                save(root/'left.json', left)
                with self.assertRaisesRegex(ValueError, 'different settings/input'):
                    run(args, corpus, queries, qrels)

    def test_input_cannot_be_output(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'source.json'
            args = SimpleNamespace(method='rerank', split='train', source=path, other=None, output=path)
            with self.assertRaisesRegex(ValueError, 'Output must differ'):
                run(args, {}, {}, {})
