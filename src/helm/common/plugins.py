from types import ModuleType
from typing import Iterable
import importlib
import pkgutil


# Copied from https://docs.python.org/3/library/pkgutil.html#pkgutil.iter_modules
def _iter_namespace(ns_pkg) -> Iterable[pkgutil.ModuleInfo]:
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def import_all_modules_in_package(module: ModuleType):
    for _, name, _ in _iter_namespace(module):
        importlib.import_module(name)
