from helm.benchmark.config_registry import register_builtin_configs_from_helm_package


def on_startup(command: str, dirty: bool):
    register_builtin_configs_from_helm_package()
