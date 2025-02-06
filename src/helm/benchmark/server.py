# mypy: check_untyped_defs = False
"""
Starts a local HTTP server to display benchmarking assets.
"""

import argparse
import importlib_resources as resources
import json
from os import path
import urllib

from bottle import Bottle, static_file, HTTPResponse
import yaml

from helm.benchmark.presentation.schema import SCHEMA_CLASSIC_YAML_FILENAME
from helm.common.general import serialize_dates


app = Bottle()


@app.get("/config.js")
def serve_config():
    if app.config["helm.release"]:
        return (
            f'window.BENCHMARK_OUTPUT_BASE_URL = "{app.config["helm.outputurl"]}";\n'
            f'window.RELEASE = "{app.config["helm.release"]}";\n'
            f'window.PROJECT_ID = "{app.config["helm.project"]}";\n'
        )
    else:
        return (
            f'window.BENCHMARK_OUTPUT_BASE_URL = "{app.config["helm.outputurl"]}";\n'
            f'window.SUITE = "{app.config["helm.suite"]}";\n'
            f'window.PROJECT_ID = "{app.config["helm.project"]}";\n'
        )


# Shim for running helm-server for old suites from old version of helm-summarize
# that do not contain schema.json.
#
# The HELM web frontend expects to find a schema.json at /benchmark_output/runs/<version>/schema.json
# which is produced by the new version of helm-summarize but not the old version.
# When serving a suite produced by the old version of helm-summarize, the schena.json will be missing.
# This shim supports those suites by serving a schena.json that is dynamically computed from schema_classic.yaml
#
# We will remove this in a few months after most users have moved to the new version of helm-summarize.
#
# TODO(2024-03-01): Remove this.
@app.get("/benchmark_output/<runs_or_releases:re:runs|releases>/<version>/schema.json")
def server_schema(runs_or_releases, version):
    relative_schema_path = path.join(runs_or_releases, version, "schema.json")
    absolute_schema_path = path.join(app.config["helm.outputpath"], relative_schema_path)
    if path.isfile(absolute_schema_path):
        response = static_file(relative_schema_path, root=app.config["helm.outputpath"])
    else:
        # Suite does not contain schema.json
        # Fall back to schema_classic.yaml from the static directory
        classic_schema_path = path.join(app.config["helm.staticpath"], SCHEMA_CLASSIC_YAML_FILENAME)
        with open(classic_schema_path, "r") as f:
            response = HTTPResponse(json.dumps(yaml.safe_load(f), indent=2, default=serialize_dates))
    response.set_header("Cache-Control", "no-cache, no-store, must-revalidate")
    response.set_header("Expires", "0")
    response.content_type = "application/json"
    return response


@app.get("/benchmark_output/<filename:path>")
def serve_benchmark_output(filename):
    response = static_file(filename, root=app.config["helm.outputpath"])
    response.set_header("Cache-Control", "no-cache, no-store, must-revalidate")
    response.set_header("Expires", "0")
    return response


@app.get("/cache/output/<filename:path>")
def serve_cache_output(filename):
    response = static_file(filename, root=app.config["helm.cacheoutputpath"])
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
        "--cache-output-path",
        type=str,
        help="The location of the filesystem cache output folder (filesystem path or URL)",
        default="prod_env/cache/output",
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

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Experimental: The name of the project to display on the landing page.",
    )
    args = parser.parse_args()

    if args.suite and args.release:
        raise ValueError("At most one of --release and --suite may be set.")

    # Determine the location of the static directory.
    # This is a hack: it assumes that the static directory has a physical location,
    # which is not always the case (e.g. when using zipimport).
    static_package_name = "helm.benchmark.static_build"
    resource_path = resources.files(static_package_name).joinpath("index.html")
    with resources.as_file(resource_path) as resource_filename:
        static_path = str(resource_filename.parent)

    app.config["helm.staticpath"] = static_path

    if urllib.parse.urlparse(args.output_path).scheme in ["http", "https"]:
        # Output path is a URL, so set the output path base URL in the frontend to that URL
        # so that the frontend reads from that URL directly.
        app.config["helm.outputpath"] = None
        # TODO: figure out helm.cacheoutputpath
        app.config["helm.outputurl"] = args.output_path
    else:
        # Output path is a location on disk, so set the output path base URL to /benchmark_output
        # and then serve files from the location on disk at that URL.
        app.config["helm.outputpath"] = path.abspath(args.output_path)
        app.config["helm.cacheoutputpath"] = path.abspath(args.cache_output_path)
        app.config["helm.outputurl"] = "benchmark_output"

    app.config["helm.suite"] = args.suite or "latest"
    app.config["helm.release"] = args.release
    app.config["helm.release"] = args.release
    app.config["helm.project"] = args.project or "lite"

    print(f"After the web server has started, go to http://localhost:{args.port} to view your website.\n")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
