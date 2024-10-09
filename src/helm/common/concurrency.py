from contextlib import AbstractContextManager
from threading import Lock
from typing import TypeVar, Generic


T = TypeVar("T")


class ThreadSafeWrapper(AbstractContextManager, Generic[T]):
    """A wrapper that makes thread-hostile objects thread-safe.

    This provides a context manager that holds a lock for accessing the inner object.

    Example usage:

        wrapped_obj = wrapper(thread_hostile_obj)
        with wrapped_obj as obj:
            # Lock is automatically held in here
            obj.do_stuff()
    """

    def __init__(self, wrapped: T):
        self._wrapped = wrapped
        self._lock = Lock()

    def __enter__(self) -> T:
        self._lock.__enter__()
        return self._wrapped

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.__exit__(exc_type, exc_value, traceback)
        pass
