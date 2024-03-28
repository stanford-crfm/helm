from abc import abstractmethod
import contextlib
import dataclasses
import json
from typing import Dict, Generator, Iterable, Mapping, Optional, Tuple

from sqlitedict import SqliteDict


class JSONEncoderWithDataclasses(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


class JSONDecoderWithDataclasses(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "__dataclass__" in obj:
            return dataclasses.replace(obj)
        return obj


def request_to_key(request: Mapping) -> str:
    """Normalize a `request` into a `key` so that we can hash using it."""
    return json.dumps(request, sort_keys=True, cls=JSONEncoderWithDataclasses)


def key_to_request(key: str, object_pairs_hook=None) -> Dict:
    """Convert a `key` back into a `request`."""
    return json.loads(key, object_pairs_hook=object_pairs_hook, cls=JSONDecoderWithDataclasses)


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
    def put(self, key: Mapping, value: Dict) -> None:
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

    def put(self, key: Mapping, value: Dict) -> None:
        key_string = request_to_key(key)
        self._sqlite_dict[key_string] = value
        self._sqlite_dict.commit()

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        for key, value in pairs:
            self.put(key, value)

    def remove(self, key: Dict) -> None:
        del self._sqlite_dict[key]
        self._sqlite_dict.commit()


class BlackHoleKeyValueStore(KeyValueStore):
    """Key value store that discards all data."""

    def __enter__(self) -> "BlackHoleKeyValueStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def contains(self, key: Dict) -> bool:
        return False

    def get(self, key: Dict) -> Optional[Dict]:
        return None

    def get_all(self) -> Generator[Tuple[Dict, Dict], None, None]:
        # Return an empty generator.
        # See: https://stackoverflow.com/a/13243870
        return
        yield

    def put(self, key: Mapping, value: Dict) -> None:
        return None

    def multi_put(self, pairs: Iterable[Tuple[Dict, Dict]]) -> None:
        return None

    def remove(self, key: Dict) -> None:
        return None
