"""One experiment runner for either dataset split."""
from importlib.metadata import version, PackageNotFoundError
from .core import (EMBEDDER, RERANKER, K, text_of, tokenize, fingerprint, read, save,
                   validate, rankings_from, sorted_items, fuse, evaluate, changes)

METHODS = ('dense', 'bm25', 'llama', 'hybrid', 'rerank')


def run(args, corpus, queries, qrels):
    source = other = None
    settings = {'dataset': 'scifact', 'split': args.split, 'method': args.method,
        'candidate_count': K, 'document_format': 'title + newline + abstract',
        'data_sha256': fingerprint({'documents': list(corpus.items()), 'queries': queries}),
        'implementation': 1}
    for flag in ('source', 'other'):
        value = getattr(args, flag)
        if value:
            if value.resolve() in {args.output.resolve(), args.output.with_name(args.output.stem + '.metrics.json').resolve()}:
                raise ValueError('Output must differ from input files.')
            loaded = rankings_from(value, args.split, queries, corpus)
            settings[flag + '_sha256'] = fingerprint(loaded)
            if flag == 'source':
                source = loaded
            else:
                other = loaded
    if args.method in ('hybrid', 'rerank') and source is None:
        raise ValueError('--source is required.')
    if args.method == 'hybrid' and other is None:
        raise ValueError('--other is required for hybrid.')
    if args.method == 'dense':
        settings.update(model=EMBEDDER, batch_size=32, search='exact cosine torch.topk')
    elif args.method == 'bm25':
        settings.update(tokenization='lowercase regex words', k1=1.5, b=0.75, epsilon=0.25)
    elif args.method == 'llama':
        settings.update(model='llama-text-embed-v2', index=args.index, namespace=args.namespace)
    elif args.method == 'hybrid':
        settings.update(rrf_constant=60, weights=[1, 1])
    elif args.method == 'rerank':
        settings.update(model=RERANKER, max_length=512, batch_size=16)
    settings['packages'] = {}
    for package in ('beir', 'sentence-transformers', 'torch', 'rank-bm25', 'pinecone'):
        try:
            settings['packages'][package] = version(package)
        except PackageNotFoundError:
            pass

    if args.output.exists():
        saved = read(args.output)
        if saved.get('settings') != settings:
            raise ValueError('Output exists with different settings/input. Choose a new --output path.')
        validate(saved['queries'], queries, corpus, complete=False)
    else:
        saved = {'settings': settings, 'queries': {}}
    pending = [qid for qid in queries if qid not in saved['queries']]
    print(f'Queries: {len(queries)}; saved: {len(saved["queries"])}; pending: {len(pending)}', flush=True)

    # Import and initialize only the method that needs computation.
    ids = list(corpus)
    texts = [text_of(corpus[doc_id]) for doc_id in ids]
    if pending and args.method == 'dense':
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(EMBEDDER)
        document_vectors = model.encode(texts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
        query_vectors = model.encode([queries[q] for q in pending], batch_size=32,
                                     convert_to_tensor=True, show_progress_bar=True)
        top = util.cos_sim(query_vectors, document_vectors).topk(k=K, dim=1)
        dense_indices, dense_scores = top.indices.tolist(), top.values.tolist()
    elif pending and args.method == 'bm25':
        from rank_bm25 import BM25Okapi
        model = BM25Okapi([tokenize(text) for text in texts], k1=1.5, b=0.75, epsilon=0.25)
    elif pending and args.method == 'llama':
        from .pinecone_io import connect, search
        index = connect(args.project, args.index)
    elif pending and args.method == 'rerank':
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANKER, max_length=512)

    for row, qid in enumerate(pending):
        if args.method == 'dense':
            items = [{'doc_id': ids[i], 'score': float(s)}
                     for i, s in zip(dense_indices[row], dense_scores[row])]
        elif args.method == 'bm25':
            items = sorted_items(ids, model.get_scores(tokenize(queries[qid])))[:K]
        elif args.method == 'llama':
            items = search(index, args.namespace, queries[qid])
        elif args.method == 'hybrid':
            items = fuse(source[qid]['top100'], other[qid]['top100'])
        else:
            candidate_ids = [item['doc_id'] for item in source[qid]['top100']]
            pairs = [(queries[qid], text_of(corpus[doc_id])) for doc_id in candidate_ids]
            scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
            items = sorted_items(candidate_ids, scores)
        record = {'top100': items}
        validate({qid: record}, queries, corpus, complete=False)
        saved['queries'][qid] = record
        save(args.output, saved)
        count = len(saved['queries'])
        if count % 50 == 0 or count == len(queries):
            print(f'Processed {count}/{len(queries)}', flush=True)

    validate(saved['queries'], queries, corpus)
    metrics = evaluate(saved['queries'], qrels)
    summary = {'settings': settings, 'query_count': len(queries), 'k_values': [10, 100],
               'ranking_policy': 'preserve saved order using ordinal scores', 'metrics': metrics}
    if source is not None:
        summary['before'] = evaluate(source, qrels)
        summary['changes'] = changes(source, saved['queries'], qrels)
    save(args.output.with_name(args.output.stem + '.metrics.json'), summary)
    for name, value in metrics.items():
        print(f'{name}: {value:.5f}')
    if 'changes' in summary:
        print('Top-10 changes:', summary['changes'])
    print('Saved:', args.output)
