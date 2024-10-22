# mypy: check_untyped_defs = False
import os
import random
import shutil
import tempfile
import requests
import socket
import subprocess
import time
from functools import partial
from multiprocessing import Pool
from dataclasses import asdict, dataclass
from typing import List

from sqlitedict import SqliteDict

from helm.common.authentication import Authentication
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.proxy.accounts import Account, set_default_quotas
from helm.proxy.services.remote_service import RemoteService
from helm.proxy.services.service import ACCOUNTS_FILE


@dataclass(frozen=True)
class TempServerInfo:
    url: str
    base_path: str


class TestRemoteServerService:
    """
    The following are our end-to-end integration tests.
    This suite of tests will bring up a rest server on an available port
    and ping different endpoints using `RemoteService`.
    """

    _ADMIN_API_KEY: str = "admin"
    _SERVER_EXECUTABLE: str = "crfm-proxy-server"
    _SERVER_TIMEOUT_SECONDS: int = 60

    @staticmethod
    def start_temp_server() -> TempServerInfo:
        def get_free_port() -> str:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # When binding a socket to port 0, the kernel will assign it a free port
            sock.bind(("", 0))
            port = str(sock.getsockname()[1])
            # Queue up to 10 requests
            sock.listen(10)
            return port

        def create_root_account() -> str:
            path: str = tempfile.mkdtemp()

            with SqliteDict(os.path.join(path, ACCOUNTS_FILE)) as cache:
                account: Account = Account(TestRemoteServerService._ADMIN_API_KEY, is_admin=True)
                set_default_quotas(account)
                cache[TestRemoteServerService._ADMIN_API_KEY] = asdict(account)
                cache.commit()
            return path

        rest_port: str = os.environ.get("TEST_PORT", get_free_port())
        url: str = f"http://127.0.0.1:{rest_port}"
        base_path: str = create_root_account()

        # Start server in a separate thread
        try:
            subprocess.Popen(
                " ".join(
                    [
                        TestRemoteServerService._SERVER_EXECUTABLE,
                        f"--port {rest_port}",
                        f"--base-path {base_path}",
                        "--workers 16",
                    ]
                ),
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"There was an error while creating the temp instance: {str(e)}")
            raise

        # Return the URL of the newly started server
        return TempServerInfo(url, base_path)

    @staticmethod
    def query(url: str, auth: Authentication, prompt: str):
        request = Request(prompt=prompt, model="simple/model1", model_deployment="simple/model1")
        response: RequestResult = RemoteService(base_url=url).make_request(auth, request)
        response_text: str = response.completions[0].text
        # With the toy model (simple/model1), we should expect the same response as the prompt
        assert response_text == prompt, f"Expected {prompt}, got back {response_text}"

    @classmethod
    def setup_class(cls):
        server_info: TempServerInfo = TestRemoteServerService.start_temp_server()
        cls.url: str = server_info.url
        cls.base_path: str = server_info.base_path
        cls.service = RemoteService(base_url=cls.url)
        cls.auth = Authentication(api_key=TestRemoteServerService._ADMIN_API_KEY)

        # Wait for the rest server (with timeout) to be brought up before running the tests.
        running = False
        start = time.time()
        while not running and time.time() - start <= TestRemoteServerService._SERVER_TIMEOUT_SECONDS:
            try:
                o = cls.service.get_general_info()
                print(o)
                running = True
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        elapsed_time_seconds = time.time() - start
        assert (
            elapsed_time_seconds <= TestRemoteServerService._SERVER_TIMEOUT_SECONDS
        ), f"elapsed: {elapsed_time_seconds}"

    @classmethod
    def teardown_class(cls):
        cls.service.shutdown(cls.auth)
        shutil.rmtree(cls.base_path)

    def test_make_request(self):
        request = Request(prompt="1 2 3", model="simple/model1", model_deployment="simple/model1")
        response: RequestResult = self.service.make_request(self.auth, request)
        assert response.success

    def test_tokenize(self):
        request = TokenizationRequest(text="1 2 3", tokenizer="simple/tokenizer1")
        response: TokenizationRequestResult = self.service.tokenize(self.auth, request)
        assert [token.value for token in response.tokens] == ["1", " ", "2", " ", "3"]

    def test_make_request_plus_sign(self):
        # Ensure + in prompt doesn't get replaced by a blank space
        request = Request(prompt="+", model="simple/model1", model_deployment="simple/model1")
        response: RequestResult = self.service.make_request(self.auth, request)
        assert response.completions[0].text == "+"
        assert response.success

    def test_update_account(self):
        account = self.service.get_account(self.auth)
        assert account.api_key == TestRemoteServerService._ADMIN_API_KEY
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

    def test_concurrent_make_request(self):
        # Fix seed for reproducibility
        random.seed(0)

        # Construct 10,000 requests with 1,000 unique prompts
        # and shuffle them to simulate simultaneous cache reads/writes.
        prompts: List[str] = [str(i) for i in range(1_000)] * 10
        random.shuffle(prompts)

        # Query in parallel with multiple processes
        with Pool(processes=16) as p:
            p.map(partial(TestRemoteServerService.query, self.url, self.auth), prompts)
