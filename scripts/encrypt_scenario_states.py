"""Encrypts prompts from scenario state.

This script modifies all scenario_state.json files in place within a suite to
encrypting all prompts, instance input text, and instance reference output text
from the `ScenarioState`s.

This is used when the scenario contains prompts that should not be displayed,
in order to reduce the chance of data leakage or to comply with data privacy
requirements.

After running this, you must re-run helm-summarize on the suite in order to
update other JSON files used by the web frontend."""

import argparse
import dataclasses
import os
import base64
from typing import Dict, Optional
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Instance, Reference
from helm.common.codec import from_json, to_json
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import Request, RequestResult
from helm.common.general import write


_SCENARIO_STATE_FILE_NAME = "scenario_state.json"
_ENCRYTED_DATA_JSON_FILE_NAME = "encryption_data.json"


class HELMEncryptor:
    def __init__(self, key, iv):
        self.key = key
        self.iv = iv
        self.encryption_data_mapping = {}
        self.idx = 0

    def encrypt_text(self, text: str) -> str:
        cipher = Cipher(algorithms.AES(self.key), modes.GCM(self.iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(text.encode()) + encryptor.finalize()
        ret_text = f"[encrypted_text_{self.idx}]"

        res = {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "key": base64.b64encode(self.key).decode(),
            "iv": base64.b64encode(self.iv).decode(),
            "tag": base64.b64encode(encryptor.tag).decode(),
        }
        assert ret_text not in self.encryption_data_mapping
        self.encryption_data_mapping[ret_text] = res
        self.idx += 1
        return ret_text


def read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)


def write_scenario_state(scenario_state_path: str, scenario_state: ScenarioState) -> None:
    with open(scenario_state_path, "w") as f:
        f.write(to_json(scenario_state))


def encrypt_reference(reference: Reference) -> Reference:
    global encryptor
    encrypted_output = dataclasses.replace(reference.output, text=encryptor.encrypt_text(reference.output.text))
    return dataclasses.replace(reference, output=encrypted_output)


def encrypt_instance(instance: Instance) -> Instance:
    global encryptor
    encrypted_input = dataclasses.replace(instance.input, text=encryptor.encrypt_text(instance.input.text))
    encrypted_references = [encrypt_reference(reference) for reference in instance.references]
    return dataclasses.replace(instance, input=encrypted_input, references=encrypted_references)


def encrypt_request(request: Request) -> Request:
    global encryptor
    return dataclasses.replace(
        request, prompt=encryptor.encrypt_text(request.prompt), messages=None, multimodal_prompt=None
    )


def encrypt_output_mapping(output_mapping: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if output_mapping is None:
        return None
    return {key: encryptor.encrypt_text(val) for key, val in output_mapping.items()}


def encrypt_result(result: Optional[RequestResult]) -> Optional[RequestResult]:
    if result is None:
        return None

    encrypted_results = [
        dataclasses.replace(completion, text=encryptor.encrypt_text(completion.text))
        for completion in result.completions
    ]
    return dataclasses.replace(result, completions=encrypted_results)


def encrypt_request_state(request_state: RequestState) -> RequestState:
    return dataclasses.replace(
        request_state,
        instance=encrypt_instance(request_state.instance),
        request=encrypt_request(request_state.request),
        output_mapping=encrypt_output_mapping(request_state.output_mapping),
        result=encrypt_result(request_state.result),
    )


def encrypt_scenario_state(scenario_state: ScenarioState) -> ScenarioState:
    encrypted_request_states = [encrypt_request_state(request_state) for request_state in scenario_state.request_states]
    return dataclasses.replace(scenario_state, request_states=encrypted_request_states)


def modify_scenario_state_for_run(run_path: str) -> None:
    scenario_state_path = os.path.join(run_path, _SCENARIO_STATE_FILE_NAME)
    scenario_state = read_scenario_state(scenario_state_path)
    encrypted_scenario_state = encrypt_scenario_state(scenario_state)
    write_scenario_state(scenario_state_path, encrypted_scenario_state)


def modify_scenario_states_for_suite(run_suite_path: str, scenario: str) -> None:
    """Load the runs in the run suite path."""
    # run_suite_path can contain subdirectories that are not runs (e.g. eval_cache, groups)
    # so filter them out.
    scenario_prefix = scenario if scenario != "all" else ""
    run_dir_names = sorted(
        [
            p
            for p in os.listdir(run_suite_path)
            if p != "eval_cache"
            and p != "groups"
            and os.path.isdir(os.path.join(run_suite_path, p))
            and p.startswith(scenario_prefix)
        ]
    )
    for run_dir_name in tqdm(run_dir_names, disable=None):
        run_path: str = os.path.join(run_suite_path, run_dir_name)
        scenario_state_path: str = os.path.join(run_path, _SCENARIO_STATE_FILE_NAME)
        if not os.path.exists(scenario_state_path):
            hwarn(f"{run_dir_name} doesn't have {_SCENARIO_STATE_FILE_NAME}, skipping")
            continue
        encryption_data_path: str = os.path.join(run_path, _ENCRYTED_DATA_JSON_FILE_NAME)
        if os.path.exists(encryption_data_path):
            hlog(f"INFO: {run_dir_name} already has {_ENCRYTED_DATA_JSON_FILE_NAME}, skipping")
            continue
        modify_scenario_state_for_run(run_path)

        # Write the encryption data to a file
        write(encryption_data_path, to_json(encryptor.encryption_data_mapping))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this encryption should go under.",
        required=True,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Name of the scenario this encryption should go under. Default is all.",
        required=True,
    )
    args = parser.parse_args()
    output_path = args.output_path
    suite = args.suite
    run_suite_path = os.path.join(output_path, "runs", suite)
    modify_scenario_states_for_suite(run_suite_path, scenario=args.scenario)


if __name__ == "__main__":
    key = os.urandom(32)  # 256-bit key
    iv = os.urandom(12)  # 96-bit IV (suitable for AES-GCM)
    encryptor = HELMEncryptor(key, iv)
    main()
