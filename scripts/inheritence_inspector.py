import os
from collections import defaultdict
import inspect
import importlib

from helm.common.object_spec import get_class_by_name

modules = [
    "helm.benchmark.scenarios." + x[:-3]
    for x in os.listdir("/home/auxy/helm/src/helm/benchmark/scenarios/")
    if x.endswith(".py")
]

for x in modules:
    importlib.import_module(x)
cls = get_class_by_name("helm.benchmark.scenarios.scenario.Scenario")

inits = {x.__name__: tuple(inspect.signature(x.__init__).parameters.keys()) for x in cls.__subclasses__()}

init_signatures = defaultdict(list)
for cls_name, signature in inits.items():
    init_signatures[signature].append(cls_name)

for signature in sorted(init_signatures.keys(), reverse=True, key=lambda key: len(init_signatures[key])):
    print(signature, len(init_signatures[signature]), init_signatures[signature], sep="|")
