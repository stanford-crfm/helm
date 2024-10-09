import json
from typing import Dict, Generator, Iterable, Mapping, Optional, Tuple

from helm.common.key_value_store import KeyValueStore
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from bson.errors import InvalidDocument
    from bson.son import SON
    from pymongo import MongoClient, ReplaceOne
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class MongoKeyValueStore(KeyValueStore):
    """Key value store backed by a MongoDB database."""

    # The number of documents to return per batch.
    _BATCH_SIZE: int = 8

    _REQUEST_KEY = "request"
    _RESPONSE_KEY = "response"

    def __init__(self, uri: str, collection_name: str):
        # TODO: Create client in __enter__ and clean up client in __exit__
        self._mongodb_client: MongoClient = MongoClient(uri)
        self._database = self._mongodb_client.get_default_database()
        self._collection = self._database.get_collection(collection_name)
        self._collection.create_index(self._REQUEST_KEY, unique=True)
        super().__init__()

    def __enter__(self) -> "MongoKeyValueStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return

    def _canonicalize_key(self, key: Mapping) -> SON:
        serialized = json.dumps(key, sort_keys=True)
        return json.loads(serialized, object_pairs_hook=SON)

    def contains(self, key: Mapping) -> bool:
        query = {self._REQUEST_KEY: self._canonicalize_key(key)}
        return self._collection.find_one(query) is not None

    def get(self, key: Mapping) -> Optional[Dict]:
        query = {self._REQUEST_KEY: self._canonicalize_key(key)}
        document = self._collection.find_one(query)
        if document is not None:
            response = document[self._RESPONSE_KEY]
            if isinstance(response, str):
                return json.loads(response)
            else:
                return response
        return None

    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        for document in self._collection.find({}).batch_size(self._BATCH_SIZE):
            request = document[self._REQUEST_KEY]
            response = document[self._RESPONSE_KEY]
            if isinstance(response, str):
                yield (request, json.loads(response))
            else:
                yield (request, response)

    def put(self, key: Mapping, value: Dict) -> None:
        request = self._canonicalize_key(key)
        document = SON([(self._REQUEST_KEY, request), (self._RESPONSE_KEY, value)])
        # The MongoDB collection should have a unique indexed on "request"
        try:
            self._collection.replace_one(filter={"request": request}, replacement=document, upsert=True)
        except (InvalidDocument, OverflowError):
            # If the document is malformed (e.g. because of null bytes in keys) or some numbers cause overflows
            # (e.g. integers exceed 8 bits) instead store the response as a string.
            alternate_document = SON([(self._REQUEST_KEY, request), (self._RESPONSE_KEY, json.dumps(value))])
            self._collection.replace_one(filter={"request": request}, replacement=alternate_document, upsert=True)

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        operations = []
        for key, value in pairs:
            request = self._canonicalize_key(key)
            document = SON([(self._REQUEST_KEY, request), (self._RESPONSE_KEY, value)])
            operations.append(ReplaceOne({self._REQUEST_KEY: request}, document, upsert=True))
        # Note: unlike put, multi_put does not support documents with null bytes in keys.
        self._collection.bulk_write(operations)

    def remove(self, key: Mapping) -> None:
        query = {self._REQUEST_KEY: self._canonicalize_key(key)}
        self._collection.delete_one(query)
