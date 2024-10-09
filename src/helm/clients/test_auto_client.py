import dataclasses
from tempfile import TemporaryDirectory
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.common.request import GeneratedOutput, Token

import pytest

from helm.common.request import Request, RequestResult
from helm.common.general import get_credentials
from helm.clients.auto_client import AutoClient


@pytest.mark.models
class TestAutoClient:
    def make_request_and_check_result(self, request, expected_result):
        credentials = get_credentials()
        if not credentials:
            pytest.skip("Skipping test because no credentials found")
        with TemporaryDirectory() as temp_dir_path:
            auto_client = AutoClient(credentials, temp_dir_path, BlackHoleCacheBackendConfig())
            actual_result = auto_client.make_request(request)
            assert actual_result.request_time or actual_result.batch_request_time
            actual_result = dataclasses.replace(
                actual_result, request_datetime=None, request_time=None, batch_request_time=None
            )
            assert expected_result == actual_result

    def test_make_request_databricks(self):
        request = Request(
            model="databricks/dolly-v2-3b",
            model_deployment="together/dolly-v2-3b",
            prompt="Elephants are one of the most",
            temperature=0.0,
            max_tokens=10,
        )
        result = RequestResult(
            success=True,
            embedding=[],
            completions=[
                GeneratedOutput(
                    text=" intelligent species on the planet. They are also one",
                    logprob=-9.087313510477543,
                    tokens=[
                        Token(
                            text="Ġintelligent",
                            logprob=-1.9816237688064575,
                        ),
                        Token(
                            text="Ġspecies",
                            logprob=-1.2881066799163818,
                        ),
                        Token(text="Ġon", logprob=-0.16092979907989502),
                        Token(text="Ġthe", logprob=-0.23620447516441345),
                        Token(
                            text="Ġplanet",
                            logprob=-0.015416033565998077,
                        ),
                        Token(text=".", logprob=-0.6683081388473511),
                        Token(text="ĠThey", logprob=-1.9231040477752686),
                        Token(text="Ġare", logprob=-0.9322243332862854),
                        Token(text="Ġalso", logprob=-0.7750787138938904),
                        Token(text="Ġone", logprob=-1.1063175201416016),
                    ],
                    finish_reason={"reason": "length"},
                )
            ],
            cached=False,
        )
        request = Request(
            model="databricks/dolly-v2-3b",
            model_deployment="together/dolly-v2-3b",
            prompt="Elephants are one of the most",
            temperature=0.0,
            max_tokens=10,
        )
        self.make_request_and_check_result(request, result)
