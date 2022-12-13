from typing import List

from helm.common.authentication import Authentication
from helm.proxy.services.server_service import ServerService
from helm.benchmark.metrics.metric_service import MetricService
from .tokenizer_service import TokenizerService


TEST_PROMPT: str = (
    "The Center for Research on Foundation Models (CRFM) is "
    "an interdisciplinary initiative born out of the Stanford "
    "Institute for Human-Centered Artificial Intelligence (HAI) "
    "that aims to make fundamental advances in the study, development, "
    "and deployment of foundation models."
)

GPT2_TEST_TOKEN_IDS: List[int] = [
    464,
    3337,
    329,
    4992,
    319,
    5693,
    32329,
    357,
    9419,
    23264,
    8,
    318,
    281,
    987,
    40625,
    10219,
    4642,
    503,
    286,
    262,
    13863,
    5136,
    329,
    5524,
    12,
    19085,
    1068,
    35941,
    9345,
    357,
    7801,
    40,
    8,
    326,
    12031,
    284,
    787,
    7531,
    14901,
    287,
    262,
    2050,
    11,
    2478,
    11,
    290,
    14833,
    286,
    8489,
    4981,
    13,
]

GPT2_TEST_TOKENS: List[str] = [
    "The",
    "ĠCenter",
    "Ġfor",
    "ĠResearch",
    "Ġon",
    "ĠFoundation",
    "ĠModels",
    "Ġ(",
    "CR",
    "FM",
    ")",
    "Ġis",
    "Ġan",
    "Ġinter",
    "disciplinary",
    "Ġinitiative",
    "Ġborn",
    "Ġout",
    "Ġof",
    "Ġthe",
    "ĠStanford",
    "ĠInstitute",
    "Ġfor",
    "ĠHuman",
    "-",
    "Cent",
    "ered",
    "ĠArtificial",
    "ĠIntelligence",
    "Ġ(",
    "HA",
    "I",
    ")",
    "Ġthat",
    "Ġaims",
    "Ġto",
    "Ġmake",
    "Ġfundamental",
    "Ġadvances",
    "Ġin",
    "Ġthe",
    "Ġstudy",
    ",",
    "Ġdevelopment",
    ",",
    "Ġand",
    "Ġdeployment",
    "Ġof",
    "Ġfoundation",
    "Ġmodels",
    ".",
]


def get_tokenizer_service(local_path: str) -> TokenizerService:
    service = ServerService(base_path=local_path, root_mode=True)
    return MetricService(service, Authentication("test"))
