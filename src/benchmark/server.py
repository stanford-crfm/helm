import http.server


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    The benchmarking assets can be served off of a static site (e.g., GitHub
    pages), but for development, it's useful to have a simple local website.
    The default http server caches, which we need to turn that off.
    Adapted from: https://stackoverflow.com/questions/42341039/remove-cache-in-a-python-http-server
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, directory="src/benchmark/static")

    def send_response_only(self, code, message=None):
        super().send_response_only(code, message)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Expires", "0")


def main():
    http.server.test(HandlerClass=MyHTTPRequestHandler)
