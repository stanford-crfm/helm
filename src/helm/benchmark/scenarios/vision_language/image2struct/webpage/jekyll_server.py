import subprocess
import os
import signal
from typing import Optional
import time
import socket
import threading

from helm.common.hierarchical_logger import hlog


class JekyllServer:
    """A class to start and stop a Jekyll server in a separate process."""

    def __init__(self, repo_path: str, port: int, verbose: bool = False):
        self.repo_path: str = repo_path
        self.verbose: bool = verbose
        self.port: int = port
        self.process: Optional[subprocess.Popen] = None
        self.success: bool = False  # Shared flag to indicate if the server started successfully

    def __del__(self):
        self.stop()
        if JekyllServer.is_port_in_use(self.port):
            if self.verbose:
                hlog(f"Port {self.port} is in use. Attempting to free it.")
            self.kill_process_using_port(self.port)
        if self.verbose:
            hlog("JekyllServer object deleted.")

    def setup_gemfile(self):
        # Check if Gemfile exists, if not, copy Gemfile.default to Gemfile
        if not os.path.exists(f"{self.repo_path}/Gemfile"):
            default_gemfile_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Gemfile.default")
            os.system(f"cp {default_gemfile_path} {self.repo_path}/Gemfile")
            if self.verbose:
                hlog("Copied Gemfile.default to Gemfile")
            return

        # Gemfile exists, check if it has the jekyll gem
        if "jekyll" in open(f"{self.repo_path}/Gemfile").read():
            # TODO: figure out if we need to do anything here
            return

        # Gemfile exists, but doesn't have jekyll gem
        with open(f"{self.repo_path}/Gemfile", "a") as file:
            file.write('gem "jekyll", "~> 4.3.3"')
            if self.verbose:
                hlog("Added jekyll gem to Gemfile")

    def setup_config(self):
        # Check if _config.yml exists, if not, copy _config.default.yml to _config.yml
        if not os.path.exists(f"{self.repo_path}/_config.yml"):
            default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "_config.default.yml")
            os.system(f"cp {default_config_path} {self.repo_path}/_config.yml")
            if self.verbose:
                hlog("Copied _config.default.yml to _config.yml")
        # Search for line starting with "port:" and replace it with "port: <port>"
        with open(f"{self.repo_path}/_config.yml", "r") as file:
            lines = file.readlines()
        with open(f"{self.repo_path}/_config.yml", "w") as file:
            for line in lines:
                if line.startswith("port"):
                    file.write(f"port: {self.port}\n")
                else:
                    file.write(line)

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a port is in use on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def kill_process_using_port(self, port: int):
        """Find and kill the process using the specified port."""
        command = f"lsof -ti:{port} | grep '[0-9]' | xargs -r kill -9"
        os.system(command)
        if self.verbose:
            hlog(f"Killed process using port {port}.")

    def stream_output(self, process: subprocess.Popen):
        """Read from stdout and stderr streams and hlog."""
        assert process.stdout is not None
        assert process.stderr is not None
        while True:
            output = process.stdout.readline()
            if not output:
                err = process.stderr.readline()
                if err:
                    decoded_line = err.decode("utf-8").strip()
                    if self.verbose:
                        hlog(f"\t> \033[91mStderr: {decoded_line}\033[0m")
                    self.success = False
                    break
                else:
                    # No more output
                    break
            else:
                decoded_line = output.decode("utf-8").strip()
                if self.verbose:
                    hlog(f"\t> Stdout: {decoded_line}")
                if "Server running... press ctrl-c to stop." in decoded_line:
                    self.success = True
                    break

    def start(self, timeout: int = 30) -> bool:
        """Start the Jekyll server in a separate process and monitor the output."""
        if JekyllServer.is_port_in_use(self.port):
            if self.verbose:
                hlog(f"Port {self.port} is in use. Attempting to free it.")
            self.kill_process_using_port(self.port)

        self.setup_gemfile()
        self.setup_config()
        command_install = f"cd {self.repo_path} && bundle install"
        subprocess.run(command_install, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        command_serve = f"cd {self.repo_path} && bundle exec jekyll serve --port {self.port}"
        self.process = subprocess.Popen(
            command_serve,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        # Start thread to read output
        output_thread = threading.Thread(target=self.stream_output, args=(self.process,))
        output_thread.start()

        # Wait for the thread to complete or timeout
        output_thread.join(timeout=timeout)

        if output_thread.is_alive():
            # If the thread is still alive after the timeout, the server did not start
            # successfully within the timeout period
            hlog("Timeout reached without detecting server start.")
            self.process.terminate()  # Terminate the process if it's still running
            output_thread.join()  # Ensure the thread is cleaned up
            return False
        else:
            if self.verbose:
                if self.success:
                    hlog("Jekyll server started successfully.")
                else:
                    hlog("Jekyll server failed to start.")
            return self.success  # Return the success flag

    def stop(self, timeout=5):
        """Stop the Jekyll server and terminate the process with a timeout.

        Args:
            timeout (int, optional): Time to wait for the server to gracefully shut down. Defaults to 5 seconds.
        """
        if self.process:
            # Try to terminate the process group gracefully
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.terminate()

            # Wait for the process to end, checking periodically
            try:
                # Wait up to `timeout` seconds for process to terminate
                for _ in range(timeout):
                    if self.process.poll() is not None:  # Process has terminated
                        break
                    time.sleep(1)  # Wait a bit before checking again
                else:
                    # If the process is still alive after the timeout, kill it
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.kill()
                    self.process.wait()  # Wait for process to be killed
                    if self.verbose:
                        hlog("Jekyll server forcefully stopped.")
            except Exception as e:
                if self.verbose:
                    hlog(f"Error stopping the Jekyll server: {e}")

            self.process = None
            if self.verbose:
                hlog("Jekyll server stopped.")
        elif self.verbose:
            hlog("Jekyll server is not running.")
