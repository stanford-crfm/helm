class OptionalDependencyNotInstalled(Exception):
    pass


def handle_module_not_found_error(e: ModuleNotFoundError):
    # TODO: Ask user to install more specific optional dependencies
    # e.g. crfm-helm[plots] or crfm-helm[server]
    raise OptionalDependencyNotInstalled(
        f"Optional dependency {e.name} is not installed. " "Please run `pip install helm-crfm[all]` to install it."
    ) from e
