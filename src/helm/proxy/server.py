"""
Starts a REST server for the frontend to interact with.
Look at `index.js` to see how the functionality is invoked.
"""

from urllib.parse import unquote_plus
import argparse
import dataclasses
import importlib_resources as resources
import json
import os
import sys
import time

from dacite import from_dict
import bottle

from helm.benchmark.config_registry import (
    register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.benchmark.model_deployment_registry import get_default_model_deployment_for_model
from helm.common.authentication import Authentication
from helm.common.cache_backend_config import CacheBackendConfig, MongoCacheBackendConfig, SqliteCacheBackendConfig
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request
from helm.common.perspective_api_request import PerspectiveAPIRequest
from helm.common.moderations_api_request import ModerationAPIRequest
from helm.common.tokenization_request import TokenizationRequest, DecodeRequest
from helm.proxy.services.service import CACHE_DIR
from helm.proxy.accounts import Account
from helm.proxy.services.server_service import ServerService
from helm.proxy.query import Query

try:
    import gunicorn  # noqa
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["proxy-server"])


bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

app = bottle.default_app()
service: ServerService


def safe_call(func, to_json=True):
    try:
        if to_json:
            bottle.response.content_type = "application/json"

        if bottle.request.method in ["DELETE", "POST", "PUT"]:
            if bottle.request.content_type == "application/json":
                params = bottle.request.json
            else:
                params = bottle.request.forms
        else:
            # bottle.request.query doesn't decode unicode properly, so do it ourselves
            params = {}
            if bottle.request.query_string != "":
                for item in bottle.request.query_string.split("&"):
                    key, value = item.split("=", 1)
                    # Replaces "+" with " " and then unquote
                    params[key] = unquote_plus(value)
        start_time = time.time()
        result = func(params)
        end_time = time.time()
        result = json.dumps(result) if to_json else result
        hlog("REQUEST {}: {} seconds, {} bytes".format(bottle.request, end_time - start_time, len(result)))
        return result
    except Exception as e:
        import traceback

        if not isinstance(e, ValueError):
            traceback.print_exc()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_str = "EXCEPTION: " + str(e) + "\n" + "\n".join(traceback.format_tb(exc_traceback))
        return json.dumps({"error": error_str}) if to_json else error_str


@app.get("/")
def handle_root():
    return bottle.redirect("/static/index.html")


@app.get("/static/<filename:path>")
def handle_static_filename(filename):
    resp = bottle.static_file(filename, root=app.config["helm.staticpath"])
    resp.add_header("Cache-Control", "no-store, must-revalidate ")
    return resp


@app.get("/output/<filename:path>")
def handle_output_filename(filename):
    resp = bottle.static_file(filename, root=app.config["crfm.proxy.outputpath"])
    return resp


@app.get("/api/general_info")
def handle_get_general_info():
    def perform(args):
        global service
        return dataclasses.asdict(service.get_general_info())

    return safe_call(perform)


@app.post("/api/account")
def handle_create_account():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        return dataclasses.asdict(service.create_account(auth))

    return safe_call(perform)


@app.delete("/api/account")
def handle_delete_account():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        api_key = args["api_key"]
        return dataclasses.asdict(service.delete_account(auth, api_key))

    return safe_call(perform)


@app.get("/api/account")
def handle_get_account():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        if "all" in args and args["all"].lower() == "true":
            return [dataclasses.asdict(account) for account in service.get_accounts(auth)]
        else:
            return [dataclasses.asdict(service.get_account(auth))]

    return safe_call(perform)


@app.put("/api/account")
def handle_update_account():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        account = from_dict(Account, json.loads(args["account"]))
        return dataclasses.asdict(service.update_account(auth, account))

    return safe_call(perform)


@app.put("/api/account/api_key")
def handle_update_api_key():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        account = from_dict(Account, json.loads(args["account"]))
        return dataclasses.asdict(service.rotate_api_key(auth, account))

    return safe_call(perform)


@app.get("/api/query")
def handle_query():
    def perform(args):
        global service
        query = Query(**args)
        return dataclasses.asdict(service.expand_query(query))

    return safe_call(perform)


@app.get("/api/request")
def handle_request():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        request = Request(**json.loads(args["request"]))
        # Hack to maintain reverse compatibility with clients with version <= 0.3.0.
        # Clients with version <= 0.3.0 do not set model_deployment, but this is now
        # required by Request.
        if not request.model_deployment:
            model_deployment = get_default_model_deployment_for_model(request.model)
            if model_deployment is None:
                raise ValueError(f"Unknown model '{request.model}'")
            request = dataclasses.replace(request, model_deployment=model_deployment)

        raw_response = dataclasses.asdict(service.make_request(auth, request))

        # Hack to maintain reverse compatibility with clients with version <= 1.0.0.
        # Clients with version <= 1.0.0 expect each token to contain a `top_logprobs`
        # field of type dict.
        for completion in raw_response["completions"]:
            for token in completion["tokens"]:
                token["top_logprobs"] = {}

        return raw_response

    return safe_call(perform)


@app.get("/api/tokenize")
def handle_tokenization():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        request = TokenizationRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.tokenize(auth, request))

    return safe_call(perform)


@app.get("/api/decode")
def handle_decode():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        request = DecodeRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.decode(auth, request))

    return safe_call(perform)


@app.get("/api/toxicity")
def handle_toxicity_request():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        request = PerspectiveAPIRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.get_toxicity_scores(auth, request))

    return safe_call(perform)


@app.get("/api/moderation")
def handle_moderation_request():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        request = ModerationAPIRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.get_moderation_results(auth, request))

    return safe_call(perform)


@app.get("/api/shutdown")
def handle_shutdown():
    def perform(args):
        global service
        auth = Authentication(**json.loads(args["auth"]))
        service.shutdown(auth)

    return safe_call(perform)


def main():
    global service
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="What port to listen on", default=1959)
    parser.add_argument("--ssl-key-file", type=str, help="Path to SSL key file")
    parser.add_argument("--ssl-cert-file", type=str, help="Path to SSL cert file")
    parser.add_argument("--ssl-ca-certs", type=str, help="Path to SSL CA certs")
    parser.add_argument("-b", "--base-path", help="What directory has credentials, etc.", default="prod_env")
    parser.add_argument("-w", "--workers", type=int, help="Number of worker processes to handle requests", default=8)
    parser.add_argument("-t", "--timeout", type=int, help="Request timeout in seconds", default=5 * 60)
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help="If non-empty, the URL of the MongoDB database that will be used for caching instead of SQLite",
        default="",
    )
    args = parser.parse_args()

    register_builtin_configs_from_helm_package()
    register_configs_from_directory(args.base_path)

    cache_backend_config: CacheBackendConfig
    if args.mongo_uri:
        cache_backend_config = MongoCacheBackendConfig(args.mongo_uri)
    else:
        sqlite_cache_path = os.path.join(args.base_path, CACHE_DIR)
        ensure_directory_exists(sqlite_cache_path)
        cache_backend_config = SqliteCacheBackendConfig(sqlite_cache_path)

    static_package_name = "helm.proxy.static"
    resource_path = resources.files(static_package_name).joinpath("index.html")
    with resources.as_file(resource_path) as resource_filename:
        static_path = str(resource_filename.parent)
    app.config["helm.staticpath"] = static_path

    service = ServerService(base_path=args.base_path, cache_backend_config=cache_backend_config)

    gunicorn_args = {
        "workers": args.workers,
        "timeout": args.timeout,
        "limit_request_line": 0,  # Controls the maximum size of HTTP request line in bytes. 0 = unlimited.
    }
    if args.ssl_key_file:
        gunicorn_args["keyfile"] = args.ssl_key_file
    if args.ssl_cert_file:
        gunicorn_args["certfile"] = args.ssl_cert_file
    if args.ssl_ca_certs:
        gunicorn_args["ca_certs"] = args.ssl_ca_certs

    # Clear arguments before running gunicorn as it also uses argparse
    sys.argv = [sys.argv[0]]
    app.config["crfm.proxy.outputpath"] = os.path.join(os.path.realpath(args.base_path), "cache", "output")
    app.run(host="0.0.0.0", port=args.port, server="gunicorn", **gunicorn_args)
