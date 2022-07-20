from common.authentication import Authentication
from proxy.service import Service
from benchmark.window_service.tokenizer_service import TokenizerService


class AdapterService(TokenizerService):
    """
    A wrapper around `Service` that makes only necessary server requests during adaptation.
    """

    def __init__(self, service: Service, auth: Authentication):
        super().__init__(service, auth)
