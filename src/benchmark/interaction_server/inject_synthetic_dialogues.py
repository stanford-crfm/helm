import argparse
import yaml
import os
import pickle
from dataclasses import replace
from sqlitedict import SqliteDict

from benchmark.adapter import InteractionRound, ScenarioState, UserInput
from common.request import RequestResult, Sequence


def main(input_path: str, runs_path: str):
    # Load synthetic dialogues
    with open(input_path, "r") as f:
        synthetic_dialogues = yaml.safe_load(f)

    # Load existing scenario state
    with open(os.path.join(runs_path, "scenario_state.pkl"), "rb") as f:
        loaded_scenario_state: ScenarioState = pickle.load(f)

    assert loaded_scenario_state.interaction_traces is not None
    assert len(loaded_scenario_state.interaction_traces) == len(synthetic_dialogues)

    new_interaction_traces = []

    for interaction_trace, synthetic_dialogue in zip(loaded_scenario_state.interaction_traces, synthetic_dialogues):

        # Create instance with synthetic prompt
        new_instance = replace(interaction_trace.instance, input=synthetic_dialogue["prompt"])

        # Start with an existing request state
        first_request_state = interaction_trace.trace[0].request_state

        # Convert yaml conversation into turn pairs
        turn_pairs = [
            {**utt1, **utt2}
            for utt1, utt2 in zip(synthetic_dialogue["dialogue"][::2], synthetic_dialogue["dialogue"][1::2])
        ]
        new_trace = []
        for tp in turn_pairs:

            # Create a new request result using bot utterance
            new_result = RequestResult(
                success=True, completions=[Sequence(text=tp["bot"], logprob=0, tokens=tp["bot"].split())], cached=True
            )
            new_request_state = replace(first_request_state, result=new_result)

            # Create a new interaction round using both bot and user text
            round = InteractionRound(request_state=new_request_state, user_input=UserInput(input=tp["user"]))
            new_trace.append(round)

        new_interaction_trace = replace(interaction_trace, instance=new_instance, trace=new_trace, trace_completed=True)
        new_interaction_traces.append(new_interaction_trace)

    # Replace interaction trace and save the new scenario state
    new_scenario_state = replace(loaded_scenario_state, interaction_traces=new_interaction_traces)
    with open(os.path.join(runs_path, "scenario_state.pkl"), "wb") as f:
        pickle.dump(new_scenario_state, f)

    # Replace interaction_traces.sqlite with the new ones
    assert new_scenario_state.interaction_traces is not None
    with SqliteDict(
        os.path.join(runs_path, "interaction_traces.sqlite"), tablename="interaction_traces", flag="n"
    ) as trace_db:
        for interaction_trace in new_scenario_state.interaction_traces:
            trace_db[str(interaction_trace._id)] = interaction_trace
        trace_db.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("input", help="Synthetic dialogues in yaml format")
    parser.add_argument("runs_path", help="Path to run")
    args = parser.parse_args()
    main(args.input, args.runs_path)
