from helm.common.critique_request import (
    CritiqueRequest,
    CritiqueRequestResult,
)
from helm.proxy.critique.critique_client import CritiqueClient
from helm.proxy.critique.mechanical_turk_critique_exporter import export_request
from helm.proxy.critique.mechanical_turk_critique_importer import import_request_result


class MechanicalTurkCritiqueClient(CritiqueClient):
    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        export_request(request)
        return import_request_result(request) or CritiqueRequestResult([])
