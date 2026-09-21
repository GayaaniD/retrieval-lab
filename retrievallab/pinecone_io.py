"""Pinecone compatibility and bounded retries in one place."""
import os
import time
from collections.abc import Mapping


def field(item, *names):
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    raise ValueError(f'Missing expected field: {names}')


def retry(operation, sleep=time.sleep):
    for attempt in range(6):
        try:
            return operation()
        except Exception as error:
            status = getattr(error, 'status_code', None) or getattr(error, 'status', None)
            if str(status) not in {'429', '500', '502', '503', '504'} or attempt == 5:
                raise
            delay = 60 if str(status) == '429' else min(2 ** (attempt + 2), 60)
            print(f'Temporary error; retrying in {delay}s.', flush=True)
            sleep(delay)


def connect(root, name):
    from dotenv import load_dotenv
    from pinecone import Pinecone
    load_dotenv(root / '.env')
    key = os.getenv('PINECONE_API_KEY')
    if not key:
        raise ValueError('Set PINECONE_API_KEY in the project .env file.')
    pc = Pinecone(api_key=key)
    description = retry(lambda: pc.describe_index(name))
    return pc.Index(host=description.host)


def search(index, namespace, query):
    response = retry(lambda: index.search(namespace=namespace,
        query={'inputs': {'text': query}, 'top_k': 100}, fields=['title']))
    if hasattr(response, 'to_dict'):
        response = response.to_dict()
    hits = field(field(response, 'result'), 'hits')
    return [{'doc_id': field(hit, 'id_', '_id', 'id'),
             'score': float(field(hit, 'score_', '_score', 'score'))} for hit in hits]


def stored_ids(index, namespace):
    # Restart listing on temporary failure; set membership removes duplicates.
    def collect():
        found = set()
        for page in index.list(namespace=namespace):
            for item in page:
                found.add(item if isinstance(item, str) else field(item, 'id', 'id_', '_id'))
        return found
    return retry(collect)
