import logging
import logging.config
import yaml
import os
import sys
import time
from typing import Any, Callable, List, Optional
from colorlog import ColoredFormatter


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

    # Far too much effort to unwind every call to hlog to go via logging,
    # And is a terrible idea to inspect the stack every time hlog is called
    # to figure out the caller,
    # So just log everything under "helm".
    logger = logging.getLogger("helm")

    def __init__(self) -> None:
        self.start_times: List[float] = []

    def indent(self) -> str:
        return "  " * len(self.start_times)

    def track_begin(self, x: Any, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self.logger.info(self.indent() + str(x) + " {", **kwargs)
        sys.stdout.flush()
        self.start_times.append(time.time())

    def track_end(self, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        t = time.time() - self.start_times.pop()
        self.logger.info(self.indent() + "} [%s]" % (format_time(t)), **kwargs)
        sys.stdout.flush()

    def log(self, x: Any, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self.logger.info(self.indent() + str(x), **kwargs)
        sys.stdout.flush()

    def debug(self, x: Any, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self.logger.debug(self.indent() + str(x), **kwargs)
        sys.stdout.flush()

    def warn(self, x: Any, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self.logger.warning(self.indent() + str(x), **kwargs)
        sys.stdout.flush()

    def error(self, x: Any, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self.logger.error(self.indent() + str(x), **kwargs)
        sys.stdout.flush()

    def exception(self, x: Any, **kwargs) -> None:
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
        self.logger.exception(self.indent() + str(x), **kwargs)
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


def hdebug(x: Any, **kwargs) -> None:
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
    singleton.debug(x, **kwargs)


def hlog(x: Any, **kwargs) -> None:
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
    singleton.log(x, **kwargs)


def hwarn(x: Any, **kwargs) -> None:
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
    singleton.warn(x, **kwargs)


def herror(x: Any, **kwargs) -> None:
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
    singleton.error(x, **kwargs)


def hexception(x: Any, **kwargs) -> None:
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
    singleton.exception(x, **kwargs)


class htrack_block:
    def __init__(self, x: Any, stacklevel=1) -> None:
        self._stacklevel = stacklevel + 1
        self.x = x

    def __enter__(self) -> None:
        singleton.track_begin(self.x, stacklevel=self._stacklevel)

    def __exit__(self, tpe: Any, value: Any, callback: Any) -> None:
        singleton.track_end(stacklevel=self._stacklevel)


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
            with htrack_block(parent + fn.__name__ + description, stacklevel=2):
                return fn(*args, **kwargs)

        return wrapper


def setup_default_logging(config_path: Optional[str] = None):
    """
    Setup Python logging for HELM

    Priority:
    1. External config file (YAML or JSON).
    2. ENV var LOG_LEVEL.
    3. a default logger to STDOUT
    """
    logger = logging.getLogger("helm")
    logger.propagate = False

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
        hdebug("setup custom HELM logging")
        return

    log_level = (os.getenv("HELM_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO").upper()
    try:
        logger.setLevel(getattr(logging, log_level))
    except AttributeError:
        logger.setLevel(logging.INFO)

    # Set formatter
    formatter: Optional[logging.Formatter] = None
    if sys.stdout.isatty():
        try:
            formatter = ColoredFormatter(
                "%(bold_black)s%(asctime)s%(reset)s %(log_color)s%(levelname)-8s%(reset)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
                reset=True,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
                style="%",
            )
        except ImportError:
            pass

    if formatter is None:
        # fallback
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    # Add default stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    hdebug("setup default HELM logging")
