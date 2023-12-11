from abc import abstractmethod
import contextlib
import json
from typing import Dict, Generator, Iterable, Optional, Tuple

from sqlitedict import SqliteDict


def request_to_key(request: Dict) -> str:
    """Normalize a `request` into a `key` so that we can hash using it."""
    return json.dumps(request, sort_keys=True)


class KeyValueStore(contextlib.AbstractContextManager):
    """Key value store that persists writes."""

    @abstractmethod
    def contains(self, key: Dict) -> bool:
        pass

    @abstractmethod
    def get(self, key: Dict) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        pass

    @abstractmethod
    def put(self, key: Dict, value: Dict) -> None:
        pass

    @abstractmethod
    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        pass

    @abstractmethod
    def remove(self, key: Dict) -> None:
        pass


class SqliteKeyValueStore(KeyValueStore):
    """Key value store backed by a SQLite file."""

    def __init__(self, path: str):
        self._sqlite_dict = SqliteDict(path)
        super().__init__()

    def __enter__(self) -> "SqliteKeyValueStore":
        self._sqlite_dict.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._sqlite_dict.__exit__(exc_type, exc_value, traceback)

    def contains(self, key: Dict) -> bool:
        return request_to_key(key) in self._sqlite_dict

    def get(self, key: Dict) -> Optional[Dict]:
        key_string = request_to_key(key)
        result = self._sqlite_dict.get(key_string)
        if result is not None:
            assert isinstance(result, dict)
            return result
        return None

    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        for key, value in self._sqlite_dict.items():
            yield (key, value)

    def put(self, key: Dict, value: Dict) -> None:
        key_string = request_to_key(key)
        self._sqlite_dict[key_string] = value
        self._sqlite_dict.commit()

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        for key, value in pairs:
            self.put(key, value)

    def remove(self, key: Dict) -> None:
        del self._sqlite_dict[key]
        self._sqlite_dict.commit()
