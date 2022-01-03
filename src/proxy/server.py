#!/usr/bin/env python

"""
Starts a REST server for the frontend to interact with.
Look at `index.js` to see how the functionality is invoked.
"""

import argparse
import bottle
import dataclasses
import json
import os
import signal
import sys
import time
from paste import httpserver

from dacite import from_dict

from common.authentication import Authentication
from common.hierarchical_logger import hlog
from common.request import Request
from proxy.accounts import Account
from .service import ServerService
from .query import Query

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

app = bottle.default_app()


def safe_call(func, to_json=True):
    try:
        if to_json:
            bottle.response.content_type = "application/json"
        if bottle.request.method in ["POST", "PUT"]:
            if bottle.request.content_type == "application/json":
                params = bottle.request.json
            else:
                params = bottle.request.forms
        else:
            params = bottle.request.query
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


@app.post("/api/account")
def handle_create_account():
    def perform(args):
        auth = Authentication(**json.loads(args["auth"]))
        return dataclasses.asdict(service.create_account(auth))

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


@app.get("/api/shutdown")
def handle_shutdown():
    def perform(args):
        pid = os.getpid()
        hlog(f"Shutting down server by killing own process {pid}...")
        hlog(os.kill(pid, signal.SIGTERM))

    return safe_call(perform)


def main():
    global service
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="What port to listen on", default=1959)
    parser.add_argument("-b", "--base-path", help="What directory has credentials, etc.", default="prod_env")
    args = parser.parse_args()

    service = ServerService(base_path=args.base_path)
    httpserver.serve(app, host="0.0.0.0", port=args.port)
