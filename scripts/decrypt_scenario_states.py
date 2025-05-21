import argparse
import dataclasses
import json
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
from helm.common.hierarchical_logger import hwarn
from helm.common.request import Request, RequestResult

_SCENARIO_STATE_FILE_NAME = "scenario_state.json"
_DECRYPTED_SCENARIO_STATE_FILE_NAME = "decrypted_scenario_state.json"
_DISPLAY_ENCRYPTION_DATA_JSON_FILE_NAME = "encryption_data.json"


class HELMDecryptor:
    def __init__(self, encryption_data_mapping: Dict[str, Dict[str, str]]):
        """
        encryption_data_mapping is a dict like:
        {
            "[encrypted_text_0]": {
                "ciphertext": "...",
                "key": "...",
                "iv": "...",
                "tag": "..."
            },
            ...
        }
        """
        self.encryption_data_mapping = encryption_data_mapping

    def decrypt_text(self, text: str) -> str:
        if text.startswith("[encrypted_text_") and text.endswith("]"):
            data = self.encryption_data_mapping.get(text)
            if data is None:
                # If not found in encryption data, return as-is or raise error
                raise ValueError(f"No decryption data found for {text}")

            ciphertext = base64.b64decode(data["ciphertext"])
            key = base64.b64decode(data["key"])
            iv = base64.b64decode(data["iv"])
            tag = base64.b64decode(data["tag"])

            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode("utf-8")
        else:
            # Not an encrypted placeholder, return as is.
            return text


def read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)


def write_scenario_state(scenario_state_path: str, scenario_state: ScenarioState) -> None:
    with open(scenario_state_path, "w") as f:
        f.write(to_json(scenario_state))


def read_encryption_data(encryption_data_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(encryption_data_path):
        raise ValueError(f"Could not load encryption data from {encryption_data_path}")
    with open(encryption_data_path) as f:
        return json.load(f)


def decrypt_reference(reference: Reference, decryptor: HELMDecryptor) -> Reference:
    decrypted_output = dataclasses.replace(reference.output, text=decryptor.decrypt_text(reference.output.text))
    return dataclasses.replace(reference, output=decrypted_output)


def decrypt_instance(instance: Instance, decryptor: HELMDecryptor) -> Instance:
    decrypted_input = dataclasses.replace(instance.input, text=decryptor.decrypt_text(instance.input.text))
    decrypted_references = [decrypt_reference(reference, decryptor) for reference in instance.references]
    return dataclasses.replace(instance, input=decrypted_input, references=decrypted_references)


def decrypt_request(request: Request, decryptor: HELMDecryptor) -> Request:
    # The encryption script sets request.messages and multimodal_prompt to None, so we don't need to decrypt them
    return dataclasses.replace(request, prompt=decryptor.decrypt_text(request.prompt))


def decrypt_output_mapping(
    output_mapping: Optional[Dict[str, str]], decryptor: HELMDecryptor
) -> Optional[Dict[str, str]]:
    if output_mapping is None:
        return None
    return {key: decryptor.decrypt_text(val) for key, val in output_mapping.items()}


def decrypt_result(result: Optional[RequestResult], decryptor: HELMDecryptor) -> Optional[RequestResult]:
    if result is None:
        return None

    decrypted_completions = [
        dataclasses.replace(completion, text=decryptor.decrypt_text(completion.text))
        for completion in result.completions
    ]
    return dataclasses.replace(result, completions=decrypted_completions)


def decrypt_request_state(request_state: RequestState, decryptor: HELMDecryptor) -> RequestState:
    return dataclasses.replace(
        request_state,
        instance=decrypt_instance(request_state.instance, decryptor),
        request=decrypt_request(request_state.request, decryptor),
        output_mapping=decrypt_output_mapping(request_state.output_mapping, decryptor),
        result=decrypt_result(request_state.result, decryptor),
    )


def decrypt_scenario_state(scenario_state: ScenarioState, decryptor: HELMDecryptor) -> ScenarioState:
    decrypted_request_states = [decrypt_request_state(rs, decryptor) for rs in scenario_state.request_states]
    return dataclasses.replace(scenario_state, request_states=decrypted_request_states)


def modify_scenario_state_for_run(run_path: str) -> None:
    scenario_state_path = os.path.join(run_path, _SCENARIO_STATE_FILE_NAME)
    encryption_data_path = os.path.join(run_path, _DISPLAY_ENCRYPTION_DATA_JSON_FILE_NAME)

    scenario_state = read_scenario_state(scenario_state_path)
    encryption_data_mapping = read_encryption_data(encryption_data_path)
    decryptor = HELMDecryptor(encryption_data_mapping)

    decrypted_scenario_state = decrypt_scenario_state(scenario_state, decryptor)
    decrypted_scenario_state_path = os.path.join(run_path, _DECRYPTED_SCENARIO_STATE_FILE_NAME)
    write_scenario_state(decrypted_scenario_state_path, decrypted_scenario_state)


def modify_scenario_states_for_suite(run_suite_path: str, scenario: str) -> None:
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
        scenario_state_path: str = os.path.join(run_suite_path, run_dir_name, _SCENARIO_STATE_FILE_NAME)
        encryption_data_path = os.path.join(run_suite_path, run_dir_name, _DISPLAY_ENCRYPTION_DATA_JSON_FILE_NAME)
        if not os.path.exists(scenario_state_path):
            hwarn(f"{run_dir_name} doesn't have {_SCENARIO_STATE_FILE_NAME}, skipping")
            continue
        if not os.path.exists(encryption_data_path):
            hwarn(f"{run_dir_name} doesn't have {_DISPLAY_ENCRYPTION_DATA_JSON_FILE_NAME}, skipping")
            continue
        run_path: str = os.path.join(run_suite_path, run_dir_name)
        modify_scenario_state_for_run(run_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this decryption should go under.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Name of the scenario this decryption should go under. Default is all.",
    )
    args = parser.parse_args()
    output_path = args.output_path
    suite = args.suite
    run_suite_path = os.path.join(output_path, "runs", suite)
    modify_scenario_states_for_suite(run_suite_path, scenario=args.scenario)


if __name__ == "__main__":
    main()
