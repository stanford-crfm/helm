import os

import sys

from benchmark.adapter import UserInput
from benchmark.executor import ExecutionSpec
from benchmark.runner import InteractiveRunner

sys.path = sys.path + ["../"]

from common.authentication import Authentication  # noqa: E402
from common.request import RequestResult  # noqa: E402
from common.general import unpickle, parse_hocon  # noqa: E402

# flake8: noqa
# An example of how to use the request API.

# TODO: replace with equivalent Adapter spec that the script for HIT creation will spit out


def load_run_spec(output_path, run_name):
    runs_path = os.path.join(output_path, "runs", run_name)
    run_spec = unpickle(os.path.join(runs_path, "run_spec.pkl"))
    return run_spec


def get_runner(args: dict) -> InteractiveRunner:
    base_path = args["base_path"]
    run_name = args["run_name"]
    output_path = args["output_path"]

    credentials_path = os.path.join(base_path, "dialogue_credentials.conf")

    if os.path.exists(credentials_path):
        with open(credentials_path) as f:
            credentials = parse_hocon(f.read())
    else:
        credentials = {}
    auth = Authentication(credentials["DIALOGUE_CREDENTIALS"])
    url = "https://crfm-models.stanford.edu"
    execution_spec = ExecutionSpec(auth=auth, url=url, parallelism=1, dry_run=False)
    run_spec = load_run_spec(output_path, run_name)
    return InteractiveRunner(execution_spec, output_path, run_spec)


def start_dialogue(args: dict):
    """
    Setup a conversation (interaction_trace) based on the provided id
    """
    # If the interaction_trace_id isn't found, this will throw an error
    # the frontend needs to handle it and display an appropriate error message
    runner: InteractiveRunner = get_runner(args)
    interaction_trace_id = args["interaction_trace_id"]
    user_id = args["user_id"]
    # Initializes the interaction_trace (after reading it from disk)
    # If it is bot_initiated it also queries the LM and returns the result
    interaction_trace = runner.initialize_interaction_trace(user_id=user_id, interaction_trace_id=interaction_trace_id)
    prompt = interaction_trace.instance.input
    utterances = []
    for round in interaction_trace.trace:
        if round.user_input is not None:
            utterances.append({"speaker": "user", "text": round.user_input.input})
        if round.request_state.result is not None:
            bot_utterance = round.request_state.result.completions[0].text.strip()
            utterances.append({"speaker": "bot", "text": bot_utterance})
    filled_surveys = [s for s in interaction_trace.surveys if s.user_id == user_id]
    survey = None
    if len(filled_surveys) > 0:
        survey = filled_surveys[0].data

    response = {
        "prompt": prompt,
        "utterances": utterances,
        "isConversationOver": interaction_trace.trace_completed,
        "survey": survey,
    }
    return response


def dialogue_turn(args: dict) -> dict:
    """
    Query LM with a user utterance. Dialogue context is obtained from the interaction_trace_id
    Returns:
        json_response: json response containing
        - bot_utterance: the bot's response to the user's utterance
    """
    user_utterance = str(args.get("user_utterance", None) or "")

    interaction_trace_id = args["interaction_trace_id"]

    runner = get_runner(args)

    # Gets the LM response and also persists the trace to disk in a sqlite file
    response: RequestResult = runner.handle_user_input(interaction_trace_id, UserInput(input=user_utterance))
    bot_utterance = response.completions[0].text

    json_response = {
        "bot_utterance": bot_utterance,
    }
    return json_response  # Outputs


def end_dialogue(args: dict) -> dict:
    """ Set trace_completed to True """
    interaction_trace_id = args["interaction_trace_id"]
    runner = get_runner(args)
    runner.terminate_interaction_trace(interaction_trace_id)
    return {"success": True}  # Outputs


def submit_interview(args: dict) -> dict:
    interaction_trace_id = args["interaction_trace_id"]
    user_id = args["user_id"]

    runner = get_runner(args)

    runner.handle_survey(user_id=user_id, interaction_trace_id=interaction_trace_id, survey=args["questions"])
    return {"success": True}  # Outputs
