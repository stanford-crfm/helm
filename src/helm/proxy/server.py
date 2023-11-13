# mypy: check_untyped_defs = False

"""
Starts a REST server for the frontend to interact with.
Look at `index.js` to see how the functionality is invoked.
"""

from urllib.parse import unquote_plus
import argparse
import dataclasses
import json
import os
import sys
import time

from dacite import from_dict
import bottle

from helm.common.authentication import Authentication
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request
from helm.common.perspective_api_request import PerspectiveAPIRequest
from helm.common.tokenization_request import TokenizationRequest, DecodeRequest
from .accounts import Account
from .services.server_service import ServerService
from .query import Query

try:
    import gunicorn  # noqa
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["proxy-server"])


bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

app = bottle.default_app()


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
    resp = bottle.static_file(filename, root=os.path.join(os.path.dirname(__file__), "static"))
    resp.add_header("Cache-Control", "no-store, must-revalidate ")
    return resp


@app.get("/api/general_info")
def handle_get_general_info():
    def perform(args):
        return dataclasses.asdict(service.get_general_info())

    return safe_call(perform)


@app.get("/api/window_service_info")
def handle_get_window_service_info():
    def perform(args):
        return dataclasses.asdict(service.get_window_service_info(args["model_name"]))

    return safe_call(perform)


@app.post("/api/account")
def handle_create_account():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        return dataclasses.asdict(service.create_account(auth))

    return safe_call(perform)


@app.delete("/api/account")
def handle_delete_account():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        api_key = args["api_key"]
        return dataclasses.asdict(service.delete_account(auth, api_key))

    return safe_call(perform)


@app.get("/api/account")
def handle_get_account():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        if "all" in args and args["all"].lower() == "true":
            return [dataclasses.asdict(account) for account in service.get_accounts(auth)]
        else:
            return [dataclasses.asdict(service.get_account(auth))]

    return safe_call(perform)


@app.put("/api/account")
def handle_update_account():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        account = from_dict(Account, json.loads(args["account"]))
        return dataclasses.asdict(service.update_account(auth, account))

    return safe_call(perform)


@app.put("/api/account/api_key")
def handle_update_api_key():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        account = from_dict(Account, json.loads(args["account"]))
        return dataclasses.asdict(service.rotate_api_key(auth, account))

    return safe_call(perform)


@app.get("/api/query")
def handle_query():
    def perform(args):
        query = Query(**args)
        return dataclasses.asdict(service.expand_query(query))

    return safe_call(perform)


@app.get("/api/request")
def handle_request():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        request = Request(**json.loads(args["request"]))
        return dataclasses.asdict(service.make_request(auth, request))

    return safe_call(perform)


@app.get("/api/tokenize")
def handle_tokenization():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        request = TokenizationRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.tokenize(auth, request))

    return safe_call(perform)


@app.get("/api/decode")
def handle_decode():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        request = DecodeRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.decode(auth, request))

    return safe_call(perform)


@app.get("/api/toxicity")
def handle_toxicity_request():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        request = PerspectiveAPIRequest(**json.loads(args["request"]))
        return dataclasses.asdict(service.get_toxicity_scores(auth, request))

    return safe_call(perform)


@app.get("/api/shutdown")
def handle_shutdown():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        service.shutdown(auth)

    return safe_call(perform)


def main():
    global service
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="What port to listen on", default=1959)
    parser.add_argument("--ssl-key-file", type=str, help="Path to SSL key file")
    parser.add_argument("--ssl-cert-file", type=str, help="Path to SSL cert file")
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

    service = ServerService(base_path=args.base_path, mongo_uri=args.mongo_uri)

    gunicorn_args = {
        "workers": args.workers,
        "timeout": args.timeout,
        "limit_request_line": 0,  # Controls the maximum size of HTTP request line in bytes. 0 = unlimited.
    }
    if args.ssl_key_file and args.ssl_cert_file:
        gunicorn_args["keyfile"] = args.ssl_key_file
        gunicorn_args["certfile"] = args.ssl_cert_file

    # Clear arguments before running gunicorn as it also uses argparse
    sys.argv = [sys.argv[0]]
    app.run(host="0.0.0.0", port=args.port, server="gunicorn", **gunicorn_args)
