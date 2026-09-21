"""Run with: uv run python -m retrievallab --help."""
import argparse
from pathlib import Path
from .core import load_dataset, rankings_from, evaluate, changes, save, text_of


def main():
    parser = argparse.ArgumentParser(description='RetrievalLab: shared retrieval and BEIR evaluation commands.')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent,
                        help='Project containing datasets/scifact, results and .env.')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Retrieve, fuse or rerank, then evaluate with BEIR.')
    run.add_argument('--method', choices=['dense', 'bm25', 'llama', 'hybrid', 'rerank'], required=True)
    run.add_argument('--source', type=Path)
    run.add_argument('--other', type=Path)
    run.add_argument('--output', type=Path)
    ev = sub.add_parser('evaluate', help='Evaluate an existing ranking file without running models.')
    ev.add_argument('--input', type=Path, required=True)
    ev.add_argument('--output', type=Path)
    compare = sub.add_parser('compare', help='Compare two saved rankings with BEIR.')
    compare.add_argument('--before', type=Path, required=True)
    compare.add_argument('--after', type=Path, required=True)
    compare.add_argument('--output', type=Path)
    inspect = sub.add_parser('inspect', help='Inspect one benchmark query and evidence ranks.')
    inspect.add_argument('--input', type=Path, required=True)
    inspect.add_argument('--query', required=True)
    verify = sub.add_parser('verify', help='Compare Pinecone record IDs with the corpus.')
    upload = sub.add_parser('upload', help='Upload corpus records absent from the namespace.')
    upload.add_argument('--execute', action='store_true', help='Write missing records; otherwise preview only.')
    for p in (run, ev, compare, inspect):
        p.add_argument('--split', choices=['train', 'test'], required=True)
    for p in (run, verify, upload):
        p.add_argument('--index', default='retrievallab-scifact-llama')
        p.add_argument('--namespace', default='scifact')
    args = parser.parse_args()
    args.project = args.project.resolve()
    # All relative file arguments are relative to --project.
    for flag in ('input', 'output', 'source', 'other', 'before', 'after'):
        value = getattr(args, flag, None)
        if value is not None and not value.is_absolute():
            setattr(args, flag, args.project / value)
    split = getattr(args, 'split', 'train')
    if args.command == 'run':
        if args.method in ('hybrid', 'rerank') and not args.source:
            parser.error('--source is required for this method.')
        if args.method == 'hybrid' and not args.other:
            parser.error('--other is required for hybrid.')
        if args.other and args.method != 'hybrid':
            parser.error('--other is only used for hybrid.')
        if args.source and args.method not in ('hybrid', 'rerank'):
            parser.error('--source is only used for hybrid or rerank.')
        if args.output is None:
            args.output = args.project / 'results' / 'refactored' / f'{args.method}_{split}.json'
    # Never overwrite a source ranking file with metric output.
    if args.command in ('evaluate', 'compare') and args.output:
        inputs = [getattr(args, f, None) for f in ('input', 'before', 'after')]
        if any(p and args.output.resolve() == p.resolve() for p in inputs):
            parser.error('--output must differ from ranking input paths.')

    corpus, queries, qrels = load_dataset(args.project, split)
    if args.command == 'run':
        from .runner import run as execute
        execute(args, corpus, queries, qrels)
    elif args.command == 'evaluate':
        rankings = rankings_from(args.input, split, queries, corpus)
        metrics = evaluate(rankings, qrels)
        for name, value in metrics.items():
            print(f'{name}: {value:.5f}')
        if args.output:
            save(args.output, {'dataset': 'scifact', 'split': split, 'query_count': len(queries),
                               'k_values': [10, 100], 'metrics': metrics})
    elif args.command == 'compare':
        before = rankings_from(args.before, split, queries, corpus)
        after = rankings_from(args.after, split, queries, corpus)
        b, a = evaluate(before, qrels), evaluate(after, qrels)
        counts = changes(before, after, qrels)
        print(f'{"Metric":<16} {"Before":>10} {"After":>10}')
        for key in b:
            print(f'{key:<16} {b[key]:>10.5f} {a[key]:>10.5f}')
        print('Top-10 changes:', counts)
        if args.output:
            save(args.output, {'split': split, 'query_count': len(queries),
                               'before': b, 'after': a, 'changes': counts})
    elif args.command == 'inspect':
        ranks = rankings_from(args.input, split, queries, corpus)
        if args.query not in queries:
            parser.error('Query ID is not in the selected split.')
        print('Query:', queries[args.query])
        items = ranks[args.query]['top100']
        relevant = {d for d, label in qrels[args.query].items() if label > 0}
        for rank, item in enumerate(items[:10], 1):
            doc_id = item['doc_id']
            label = 'LABELLED RELEVANT' if doc_id in relevant else 'Not labelled relevant'
            print(f'{rank}. {doc_id} | {item["score"]:.4f} | {label}\n{corpus[doc_id]["title"]}')
        positions = {item['doc_id']: n for n, item in enumerate(items, 1)}
        for doc_id in sorted(relevant):
            print('Evidence:', doc_id, '| Rank:', positions.get(doc_id, 'Outside saved top 100'))
    else:
        from .pinecone_io import connect, stored_ids, retry
        index = connect(args.project, args.index)
        present = stored_ids(index, args.namespace)
        missing, extra = set(corpus) - present, present - set(corpus)
        print(f'Expected: {len(corpus)}; stored: {len(present)}; missing: {len(missing)}; unexpected: {len(extra)}')
        if args.command == 'verify':
            if missing or extra:
                print('Missing IDs (first 20):', sorted(missing)[:20])
                print('Unexpected IDs (first 20):', sorted(extra)[:20])
                raise SystemExit(1)
            print('PASS: all expected IDs are present.')
        elif not args.execute:
            print('Preview only. Use --execute to upload missing IDs. Existing IDs are not updated.')
        else:
            missing_ids = [d for d in corpus if d in missing]
            for start in range(0, len(missing_ids), 32):
                batch = missing_ids[start:start + 32]
                records = [{'_id': d, 'text': text_of(corpus[d]), 'title': corpus[d]['title']} for d in batch]
                retry(lambda: index.upsert_records(namespace=args.namespace, records=records))
                print(f'Accepted {start + len(batch)}/{len(missing_ids)} missing records.', flush=True)
            print('Upload complete; run verify after indexing becomes visible.')


if __name__ == '__main__':
    main()
