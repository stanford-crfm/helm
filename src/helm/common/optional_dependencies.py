from typing import List, Optional


class OptionalDependencyNotInstalled(Exception):
    pass


def handle_module_not_found_error(e: ModuleNotFoundError, suggestions: Optional[List[str]] = None):
    # TODO: Ask user to install more specific optional dependencies
    # e.g. crfm-helm[plots] or crfm-helm[server]
    suggested_commands = " or ".join(
        [f"`pip install crfm-helm[{suggestion}]`" for suggestion in (suggestions or []) + ["all"]]
    )
    raise OptionalDependencyNotInstalled(
        f"Optional dependency {e.name} is not installed. Please run {suggested_commands} to install it."
    ) from e
