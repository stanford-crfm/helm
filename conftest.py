from helm.benchmark.config_registry import register_builtin_configs_from_helm_package


def pytest_sessionstart(session):
    register_builtin_configs_from_helm_package()
