# mypy: check_untyped_defs = False
"""
Starts a local HTTP server to display benchmarking assets.
"""

import argparse
import importlib_resources as resources
from os import path
import urllib

from bottle import Bottle, static_file


app = Bottle()


@app.get("/config.js")
def serve_config():
    if app.config["helm.release"]:
        return (
            f'window.BENCHMARK_OUTPUT_BASE_URL = "{app.config["helm.outputurl"]}";\n'
            f'window.RELEASE = "{app.config["helm.release"]}";\n'
        )
    else:
        return (
            f'window.BENCHMARK_OUTPUT_BASE_URL = "{app.config["helm.outputurl"]}";\n'
            f'window.SUITE = "{app.config["helm.suite"]}";\n'
        )


@app.get("/benchmark_output/<filename:path>")
def serve_benchmark_output(filename):
    response = static_file(filename, root=app.config["helm.outputpath"])
    response.set_header("Cache-Control", "no-cache, no-store, must-revalidate")
    response.set_header("Expires", "0")
    return response


@app.get("/")
@app.get("/<filename:path>")
def serve_static(filename="index.html"):
    response = static_file(filename, root=app.config["helm.staticpath"])
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="What port to listen on", default=8000)
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="The location of the output path (filesystem path or URL)",
        default="benchmark_output",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Name of the suite to serve (default is latest).",
    )
    parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Experimental: The release to serve. If unset, don't serve a release, and serve the latest suite instead.",
    )
    args = parser.parse_args()

    if args.suite and args.release:
        raise ValueError("At most one of --release and --suite may be set.")

    # Determine the location of the static directory.
    # This is a hack: it assumes that the static directory has a physical location,
    # which is not always the case (e.g. when using zipimport).
    resource_path = resources.files("helm.benchmark.static").joinpath("index.html")
    with resources.as_file(resource_path) as resource_filename:
        static_path = str(resource_filename.parent)

    app.config["helm.staticpath"] = static_path

    if urllib.parse.urlparse(args.output_path).scheme in ["http", "https"]:
        # Output path is a URL, so set the output path base URL in the frontend to that URL
        # so that the frontend reads from that URL directly.
        app.config["helm.outputpath"] = None
        app.config["helm.outputurl"] = args.output_path
    else:
        # Output path is a location on disk, so set the output path base URL to /benchmark_output
        # and then serve files from the location on disk at that URL.
        app.config["helm.outputpath"] = path.abspath(args.output_path)
        app.config["helm.outputurl"] = "benchmark_output"

    app.config["helm.suite"] = args.suite or "latest"
    app.config["helm.release"] = args.release

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
