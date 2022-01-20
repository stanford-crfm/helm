import os
import requests
import socket
import subprocess
import time

from common.authentication import Authentication

from common.request import Request
from .remote_service import RemoteService


class TestRemoteServerService:
    """
    The following are our end-to-end integration tests.
    This suite of tests will bring up a rest server on an available port
    and ping different endpoints using `RemoteService`.
    """

    _SERVER_EXECUTABLE: str = "venv/bin/proxy-server"
    _SERVER_TIMEOUT_SECONDS: int = 60

    @staticmethod
    def start_temp_server() -> str:
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
                        TestRemoteServerService._SERVER_EXECUTABLE,
                        f"--port {rest_port}",
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

    @classmethod
    def setup_class(cls):
        cls.url = TestRemoteServerService.start_temp_server()
        cls.service = RemoteService(base_url=cls.url)
        cls.auth = Authentication(api_key="crfm")

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

    def test_make_request(self):
        request = Request(prompt="1 2 3", model="simple/model1")
        response = self.service.make_request(self.auth, request)
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
