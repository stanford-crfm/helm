from common.authentication import Authentication
from proxy.server_service import ServerService
from benchmark.window_service.tokenizer_service import TokenizerService
from benchmark.metric_service import MetricService


def get_tokenizer_service(local_path: str) -> TokenizerService:
    service = ServerService(base_path=local_path, root_mode=True)
    return MetricService(service, Authentication("test"))
