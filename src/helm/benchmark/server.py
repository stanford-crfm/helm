"""
Starts a local HTTP server to display benchmarking assets.
"""

import argparse
import importlib_resources as resources
from os import path

from bottle import Bottle, static_file


app = Bottle()


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
    global service
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="What port to listen on", default=8000)
    parser.add_argument(
        "-o", "--output-path", type=str, help="The location of the output path", default="benchmark_output"
    )
    args = parser.parse_args()

    # Determine the location of the static directory.
    # This is a hack: it assumes that the static directory has a physical location,
    # which is not always the case (e.g. when using zipimport).
    resource_path = resources.files("helm.benchmark.static").joinpath("index.html")
    with resources.as_file(resource_path) as resource_filename:
        static_path = str(resource_filename.parent)

    app.config["helm.staticpath"] = static_path
    app.config["helm.outputpath"] = path.abspath(args.output_path)

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
