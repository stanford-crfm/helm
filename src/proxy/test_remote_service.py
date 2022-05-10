import os
from typing import List

import random
import requests
import socket
import subprocess
import time
from multiprocessing import Pool

from common.authentication import Authentication
from common.request import Request, RequestResult
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from .remote_service import RemoteService

SERVER_EXECUTABLE: str = "venv/bin/proxy-server"
SERVER_TIMEOUT_SECONDS: int = 60


def start_temp_server(num_workers: int = 1) -> str:
    def get_free_port() -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # When binding a socket to port 0, the kernel will assign it a free port
        sock.bind(("", 0))
        port = str(sock.getsockname()[1])
        # Queue up to 10 requests
        sock.listen(10)
        return port

    rest_port: str = os.environ.get("TEST_PORT", get_free_port())
    url: str = f"http://127.0.0.1:{rest_port}"

    # Start server in a separate thread
    try:
        subprocess.Popen(
            " ".join(
                [
                    SERVER_EXECUTABLE,
                    f"--port {rest_port}",
                    f"--workers {num_workers}",
                    "--base-path test_env",
                    "--read-only",
                ]
            ),
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"There was an error while creating the temp instance: {str(e)}")
        raise

    # Return the URL of the newly started server
    return url


def await_server_start(service: RemoteService):
    """Wait for the rest server (with timeout) to be brought up before running the tests."""
    running: bool = False
    start: float = time.time()
    while not running and time.time() - start <= SERVER_TIMEOUT_SECONDS:
        try:
            print(service.get_general_info())
            running = True
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    elapsed_time_seconds: float = time.time() - start
    assert (
        elapsed_time_seconds <= SERVER_TIMEOUT_SECONDS
    ), f"Elapsed: {elapsed_time_seconds} seconds while waiting for server to be brought up."


class TestRemoteServerService:
    """
    The following are our end-to-end integration tests.
    This suite of tests will bring up a rest server on an available port
    and ping different endpoints using `RemoteService`.
    """

    @classmethod
    def setup_class(cls):
        cls.url = start_temp_server()
        cls.service = RemoteService(base_url=cls.url)
        cls.auth = Authentication(api_key="crfm")
        await_server_start(cls.service)

    @classmethod
    def teardown_class(cls):
        cls.service.shutdown(cls.auth)

    def test_make_request(self):
        request = Request(prompt="1 2 3", model="simple/model1")
        response: RequestResult = self.service.make_request(self.auth, request)
        assert response.success

    def test_tokenize(self):
        request = TokenizationRequest(text="1 2 3", model="simple/model1")
        response: TokenizationRequestResult = self.service.tokenize(self.auth, request)
        assert [token.text for token in response.tokens] == ["1", "2", "3"]

    def test_make_request_plus_sign(self):
        # Ensure + in prompt doesn't get replaced by a blank space
        request = Request(prompt="+", model="simple/model1")
        response: RequestResult = self.service.make_request(self.auth, request)
        assert response.completions[0].text == "+"
        assert response.success

    def test_update_account(self):
        account = self.service.get_account(self.auth)
        assert account.api_key == "crfm"
        account.description = "new description"
        account = self.service.update_account(self.auth, account)
        assert account.description == "new description"

    def test_create_account(self):
        self.service.create_account(self.auth)
        accounts = self.service.get_accounts(self.auth)
        assert len(accounts) == 2

    def test_rotate_api_key(self):
        account = self.service.create_account(self.auth)
        old_api_key = account.api_key
        account = self.service.rotate_api_key(self.auth, account)
        assert account.api_key != old_api_key


class TestRemoteServerServiceWithMultipleWorkers:
    def setup_method(self):
        self.url = start_temp_server(num_workers=16)
        self.service = RemoteService(base_url=self.url)
        self.auth = Authentication(api_key="crfm")
        await_server_start(self.service)

    def teardown_method(self):
        self.service.shutdown(self.auth)

    def query(self, prompt: str):
        request = Request(prompt=prompt, model="simple/model1")
        response: RequestResult = RemoteService(base_url=self.url).make_request(self.auth, request)
        response_text: str = response.completions[0].text
        # With the toy model, we should expect the same response as the prompt
        assert response_text == prompt, f"Expected {prompt}, got back {response_text}"

    def test_concurrent_make_request(self):
        # Fix seed for reproducibility
        random.seed(0)

        # Construct 10,000 requests with 1,000 unique prompts
        # and shuffle them to simulate simultaneous cache reads/writes.
        prompts: List[str] = [str(i) for i in range(1_000)] * 10
        random.shuffle(prompts)

        # Query in parallel with multiple processes
        with Pool(processes=16) as p:
            p.map(self.query, prompts)
