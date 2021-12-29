import sys
import time
from typing import Any, Callable, List, Optional


class HierarchicalLogger(object):
    """
    A hierarchical logger that tracks the execution flow of the code, along
    with how long we're spending in each block.  Usage:

        @htrack(None)
        def compute_stuff(...):
            with htrack_block('Training'):
                hlog('something')

    Output:

        Training {
          something
        } [0s]
    """

    def __init__(self) -> None:
        self.start_times: List[float] = []

    def indent(self) -> str:
        return "  " * len(self.start_times)

    def track_begin(self, x: Any) -> None:
        print(self.indent() + str(x) + " {")
        sys.stdout.flush()
        self.start_times.append(time.time())

    def track_end(self) -> None:
        t = time.time() - self.start_times.pop()
        print(self.indent() + "} [%s]" % (format_time(t)))
        sys.stdout.flush()

    def log(self, x: Any) -> None:
        print(self.indent() + str(x))
        sys.stdout.flush()


def format_time(s: float) -> str:
    """Return a nice string representation of `s` seconds."""
    m = int(s / 60)
    s -= m * 60
    h = int(m / 60)
    m -= h * 60
    s = int(s * 1000) / 1000.0
    return ("" if h == 0 else str(h) + "h") + ("" if m == 0 else str(m) + "m") + (str(s) + "s")


singleton = HierarchicalLogger()

############################################################
# Exposed public methods


def hlog(x: Any) -> None:
    singleton.log(x)


class htrack_block:
    def __init__(self, x: Any) -> None:
        self.x = x

    def __enter__(self) -> None:
        singleton.track_begin(self.x)

    def __exit__(self, tpe: Any, value: Any, callback: Any) -> None:
        singleton.track_end()


class htrack:
    """
    Open a track to call the annotated function.
    Display `spec`, which is a string, where $0, $1, $2 stand in for the various arguments to the function.
    For example:

        @htrack("a=$1, b=$2"):
        def compute(self, a, b):
            ...
    """

    def __init__(self, spec: Optional[str]) -> None:
        self.spec: Optional[str] = spec

    def __call__(self, fn: Callable) -> Any:
        def wrapper(*args, **kwargs):  # type:ignore
            if len(args) > 0 and hasattr(args[0], fn.__name__):
                parent = type(args[0]).__name__ + "."
            else:
                parent = ""
            if self.spec is not None:
                description = ": " + self.spec
                for i, v in enumerate(args):
                    description = description.replace("$" + str(i), str(v))
                for k, v in kwargs.items():
                    description = description.replace("$" + k, str(v))
            else:
                description = ""
            with htrack_block(parent + fn.__name__ + description):
                return fn(*args, **kwargs)

        return wrapper
