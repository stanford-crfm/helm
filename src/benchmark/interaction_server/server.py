import sys
import argparse
from flask import Flask, request, render_template

from common.general import ensure_directory_exists
import dialogue_interface

app = Flask(__name__)

file_globals: dict = {}

class CustomFlask(Flask):
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        variable_start_string='%%',  # Default is '{{', I'm changing this because Vue.js uses '{{' / '}}'
        variable_end_string='%%',
    ))


app = CustomFlask(__name__)  # This replaces your existing "app = Flask(__name__)"

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/dialogue/interface")
def template_test():
    return render_template('dialogue/interface.html')

@app.route("/api/dialogue/start", methods=["POST"])
def start_dialogue():
    args = request.json
    args["output_path"] = file_globals["output_path"]
    args["base_path"] = file_globals["base_path"]
    response = dialogue_interface.start_dialogue(args)
    return response


@app.route("/api/dialogue/conversation", methods=["POST"])
def handle_utterance():
    args = request.json
    args["output_path"] = file_globals["output_path"]
    args["base_path"] = file_globals["base_path"]
    response = dialogue_interface.dialogue_turn(args)
    return response


@app.route("/api/dialogue/end", methods=["POST"])
def end_dialogue():
    args = request.json
    args["output_path"] = file_globals["output_path"]
    args["base_path"] = file_globals["base_path"]
    response = dialogue_interface.end_dialogue(args)
    return response


@app.route("/api/dialogue/interview", methods=["POST"])
def submit_interview():
    args = request.json
    args["output_path"] = file_globals["output_path"]
    args["base_path"] = file_globals["base_path"]
    response = dialogue_interface.submit_interview(args)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="What port to listen on", default=80)
    parser.add_argument("-b", "--base-path", help="What directory has credentials, etc.", default="src/benchmark/interaction_server/interaction_env")
    parser.add_argument(
        "-o", "--output-path", help="What directory stores interactive scenarios", default="benchmark_output"
    )
    parser.add_argument(
        "-r", "--read-only", action="store_true", help="To start a read-only service (for testing and debugging)."
    )
    args = parser.parse_args()
    ensure_directory_exists(args.output_path)
    file_globals["output_path"] = args.output_path
    file_globals["base_path"] = args.base_path
    app.run(host="0.0.0.0", port=args.port) # Use 0.0.0.0 on GCP, 127.0.0.1 elsewhere


if __name__ == "__main__":
    main()
