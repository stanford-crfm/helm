from collections import Counter
from dacite import from_dict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import argparse
import json
import os
import shutil
import time

from diffusers import DiffusionPipeline
import torch

from src.helm.common.cache import (
    KeyValueStore,
    KeyValueStoreCacheConfig,
    MongoCacheConfig,
    SqliteCacheConfig,
    create_key_value_store,
)
from src.helm.common.request import Request
from src.helm.common.file_caches.local_file_cache import LocalFileCache
from src.helm.common.general import ensure_directory_exists
from src.helm.common.hierarchical_logger import hlog, htrack_block
from src.helm.proxy.clients.deep_floyd_client import DeepFloydClient


"""
Launches Slurm jobs to generate images for requests that have not been cached for the three DeepFloyd
models (M, L, and XL).


Example usage (after a dryrun with a run suite named "deepfloyd"):

python3 scripts/offline_eval/deepfloyd/deepfloyd_slurm.py launch --run-suite-path benchmark_output/runs/deepfloyd \
--mongo-uri <MongoDB address>

-----

More information about DeepFloyd:

From  https://huggingface.co/docs/diffusers/main/en/api/pipelines/if#text-to-image-generation

DeepFloyd IF is a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and
language understanding. The model is a modular composed of a frozen text encoder and three cascaded pixel
diffusion modules:

Stage 1: a base model that generates 64x64 px image based on text prompt
Stage 2: a 64x64 px => 256x256 px super-resolution model
Stage 3: a 256x256 px => 1024x1024 px super-resolution model Stage 1 and Stage 2 utilize a frozen text encoder
based on the T5 transformer to extract text embeddings, which are then fed into a UNet architecture enhanced with
cross-attention and attention pooling. Stage 3 is Stability’s x4 Upscaling model. The result is a highly
efficient model that outperforms current state-of-the-art models, achieving a zero-shot FID score of 6.66 on the
COCO dataset. Our work underscores the potential of larger UNet architectures in the first stage of cascaded
diffusion models and depicts a promising future for text-to-image synthesis.

conda activate deepfloyd

Minimum set of dependencies:
accelerate~=0.19.0
dacite~=1.6.0
diffusers[torch]~=0.16.1
pyhocon~=0.3.59
pymongo~=4.2.0
retrying~=1.3.3
safetensors~=0.3.1
sentencepiece~=0.1.97
sqlitedict~=1.7.0
tqdm~=4.64.1
transformers~=4.29.2
zstandard~=0.18.0

"""

ORGANIZATION: str = "DeepFloyd"

# Modes
LAUNCH_MODE: str = "launch"
INFERENCE_MODE: str = "inference"


class DeepFloyd:
    MODEL_NAME_TO_MODELS: Dict[str, Tuple[str, str]] = {
        "IF-I-XL-v1.0": ("DeepFloyd/IF-I-XL-v1.0", "DeepFloyd/IF-II-L-v1.0"),  # XL
        "IF-I-L-v1.0": ("DeepFloyd/IF-I-L-v1.0", "DeepFloyd/IF-II-L-v1.0"),  # Large
        "IF-I-M-v1.0": ("DeepFloyd/IF-I-M-v1.0", "DeepFloyd/IF-II-M-v1.0"),  # Medium
    }

    @staticmethod
    def initialize_model(stage1_model_name: str, stage2_model_name: str):
        with htrack_block(f"Initializing the three stages of the IF model: {stage1_model_name}"):
            # stage 1
            stage_1 = DiffusionPipeline.from_pretrained(stage1_model_name, torch_dtype=torch.float16)
            stage_1.enable_model_cpu_offload()

            # stage 2
            stage_2 = DiffusionPipeline.from_pretrained(stage2_model_name, text_encoder=None, torch_dtype=torch.float16)
            stage_2.enable_model_cpu_offload()

            # stage 3
            safety_modules = {
                "feature_extractor": stage_1.feature_extractor,
                "safety_checker": stage_1.safety_checker,
                "watermarker": stage_1.watermarker,
            }
            stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
            )
            stage_3.enable_model_cpu_offload()
            return stage_1, stage_2, stage_3

    def __init__(self, model_name: str, file_cache_path: str, key_value_cache_config: KeyValueStoreCacheConfig):
        stage1_model, stage2_model = self.MODEL_NAME_TO_MODELS[model_name]
        self._model_engine: str = model_name
        self._stage_1, self._stage_2, self._stage_3 = self.initialize_model(stage1_model, stage2_model)

        self._file_cache = LocalFileCache(file_cache_path, "png")
        self._key_value_cache_config: KeyValueStoreCacheConfig = key_value_cache_config

    def _run_inference_single_image(self, prompt: str, file_path: str, seed: int) -> None:
        # Generating text embeddings
        prompt_embeds, negative_embeds = self._stage_1.encode_prompt(prompt)

        generator = torch.manual_seed(seed)
        image = self._stage_1(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images

        image = self._stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
        ).images

        image = self._stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
        image[0].save(file_path)

    def _process_request(self, raw_request: Dict, store: KeyValueStore):
        assert not store.contains(raw_request), "Request was found in the cache."

        image_paths: List[str] = []
        start_time: float = time.time()
        for i in range(raw_request["n"]):
            file_path: str = self._file_cache.get_unique_file_location()
            self._run_inference_single_image(raw_request["prompt"], file_path, i)
            image_paths.append(file_path)
        total_inference_time: float = time.time() - start_time

        result: Dict = {"images": image_paths, "total_inference_time": total_inference_time}
        store.put(raw_request, result)

    def run_inference(self, requests_path: str):
        """Runs inference for all the requests."""
        with create_key_value_store(self._key_value_cache_config) as store:
            with open(requests_path, "r") as f:
                for line in tqdm(f.readlines()):
                    raw_request: dict = json.loads(line)["request"]
                    self._process_request(raw_request, store)


class SlurmRunner:
    DEFAULT_NLP_RUN_ARGS: str = (
        "-a deepfloyd -c 4 --memory 64g -w /nlp/scr4/nlp/crfm/benchmarking/benchmarking"
        " -g 1 --exclude jagupard[10-31],sphinx3"
    )

    def __init__(self, run_suite_path: str, key_value_cache_config: KeyValueStoreCacheConfig):
        assert os.path.exists(run_suite_path), f"A run suite path does not exist at {run_suite_path}"
        self._run_suite_path: str = run_suite_path

        self._key_value_cache_config: KeyValueStoreCacheConfig = key_value_cache_config

        # Output path which contains the requests and log files
        self._base_path: str = f"{ORGANIZATION}_output"
        if os.path.exists(self._base_path):
            hlog(f"Found an existing output folder at {self._base_path}. Removing...")
            shutil.rmtree(self._base_path)
        ensure_directory_exists(self._base_path)

        self._logs_path: str = os.path.join(self._base_path, "logs")
        ensure_directory_exists(self._logs_path)

    def launch(
        self,
        shard_size: int = 1000,
        machine: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Given a run suite folder:
        1. Goes through all the valid run folders.
        2. Gathers requests that are not cached.
        3. Writes out `shard_size` requests to a single file for a given model until
           all the non-cached requests are exhausted.
        4. Launches a Slurm job to run inference for the specific model size on a given shard.
           If `dry_run` is True, Just outputs the necessary files.
        """
        model_to_counts: Dict[str, Counter] = {
            model: Counter(inference_count=0, cached_count=0) for model in DeepFloyd.MODEL_NAME_TO_MODELS.keys()
        }
        model_to_requests_to_process: Dict[str, List[Dict]] = {
            model: [] for model in DeepFloyd.MODEL_NAME_TO_MODELS.keys()
        }

        # Go through all the valid run folders, pull requests from the scenario_state.json
        with create_key_value_store(self._key_value_cache_config) as store:
            for run_dir in tqdm(os.listdir(self._run_suite_path)):
                run_path: str = os.path.join(self._run_suite_path, run_dir)

                if not os.path.isdir(run_path):
                    continue

                with htrack_block(f"Processing run directory: {run_dir}"):
                    scenario_state_path: str = os.path.join(run_path, "scenario_state.json")

                    if not os.path.isfile(scenario_state_path):
                        hlog(
                            f"{run_dir} is missing a scenario_state.json file. Expected at path: {scenario_state_path}."
                        )
                        continue

                    with open(scenario_state_path) as scenario_state_file:
                        scenario_state = json.load(scenario_state_file)
                        model_name: str = scenario_state["adapter_spec"]["model"]
                        current_model_engine: str = model_name.split("/")[-1]

                        if current_model_engine not in DeepFloyd.MODEL_NAME_TO_MODELS.keys():
                            continue

                        counts = model_to_counts[current_model_engine]
                        requests_to_process: List[Dict] = model_to_requests_to_process[current_model_engine]
                        for request_state in tqdm(scenario_state["request_states"]):
                            request: Request = from_dict(Request, request_state["request"])
                            raw_request: Dict = DeepFloydClient.convert_to_raw_request(request)
                            if store.contains(raw_request):
                                counts["cached_count"] += 1
                            else:
                                counts["inference_count"] += 1
                                requests_to_process.append(raw_request)

        # Launch jobs on Slurm for each models
        dry_run_path: str = os.path.join(self._logs_path, "dryrun.log")
        for model_name, requests_to_process in model_to_requests_to_process.items():
            hlog(f"Launching jobs for {model_name}")
            job_name_to_requests_path: Dict[str, str] = dict()

            shard: int = 0
            job_name: str = f"{model_name}_shard{shard}"

            # Shard and write out requests to files
            for i, request in enumerate(requests_to_process):
                if i % shard_size == 0:
                    shard += 1
                    job_name = f"{model_name}_shard{shard}"

                requests_path: str = os.path.join(self._base_path, f"{job_name}.jsonl")
                with open(requests_path, "a") as out_file:
                    request_json: str = json.dumps({"request": request}, sort_keys=True, ensure_ascii=False)
                    out_file.write(request_json + "\n")
                job_name_to_requests_path[job_name] = requests_path

            # Launch a job for each shard
            for job_name, requests_path in job_name_to_requests_path.items():
                nlp_run_args: str = self.DEFAULT_NLP_RUN_ARGS
                if machine is not None:
                    nlp_run_args += f" -m {machine}"

                    if "sphinx" in machine:
                        nlp_run_args += " -p high "

                log_path: str = os.path.join(self._logs_path, f"{job_name}.log")
                command: str = (
                    f"nlprun {nlp_run_args} --job-name {job_name} 'python3 scripts/offline_eval/deepfloyd/"
                    f"deepfloyd_slurm.py {INFERENCE_MODE} --model {model_name} --requests-path {requests_path} "
                    f"--mongo-uri {mongo_uri} > {log_path} 2>&1'"
                )
                hlog(command)

                if dry_run:
                    with open(dry_run_path, "a") as dryrun_file:
                        dryrun_file.write(f"{command}\n")
                else:
                    os.system(command)

        # Print out the processing stats for each model size
        for model_name, counts in model_to_counts.items():
            hlog(
                f"{model_name}: Will process {counts['inference_count']} requests. "
                f"{counts['cached_count']} requests already had entries in the cache."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="prod_env/cache", help="Path to the cache directory")
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help=(
            "For a MongoDB cache, Mongo URI to copy items to. "
            "Example format: mongodb://[username:password@]host1[:port1]/dbname"
        ),
    )
    parser.add_argument("--model", type=str, help="Name of the model", choices=DeepFloyd.MODEL_NAME_TO_MODELS.keys())
    parser.add_argument("--requests-path", type=str, help="Path to the requests")
    parser.add_argument("--run-suite-path", type=str, help="Path to run path.")
    parser.add_argument("--shard-size", type=int, default=1000, help="How many requests each Slurm job is processing")
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="If specified, runs on that specific machine",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=None,
        help="Skips launching jobs. Only logs and generates output files.",
    )

    parser.add_argument("mode", type=str, help="Which mode", choices=[LAUNCH_MODE, INFERENCE_MODE])
    args = parser.parse_args()

    # Validate args
    cache_config: KeyValueStoreCacheConfig
    if args.mongo_uri:
        hlog(f"Initialized MongoDB cache with URI: {args.mongo_uri}")
        cache_config = MongoCacheConfig(args.mongo_uri, ORGANIZATION)
    elif args.cache_dir:
        hlog(f"WARNING: Initialized SQLite cache at path: {args.cache_dir}. Are you debugging??")
        cache_config = SqliteCacheConfig(os.path.join(args.cache_dir, f"{ORGANIZATION}.sqlite"))
    else:
        raise ValueError("Either --cache-dir or --mongo-uri should be specified")

    if args.mode == LAUNCH_MODE:
        assert args.run_suite_path is not None, "--run-suite-path must be specified when launching inference jobs"
        slurm_runner = SlurmRunner(run_suite_path=args.run_suite_path, key_value_cache_config=cache_config)
        slurm_runner.launch(
            shard_size=args.shard_size, machine=args.machine, mongo_uri=args.mongo_uri, dry_run=args.dry_run
        )
    elif args.mode == INFERENCE_MODE:
        assert args.model is not None, "--model must be specified when running inference"
        assert args.requests_path is not None, "--requests-path must be specified when running inference"
        deep_floyd = DeepFloyd(
            model_name=args.model,
            file_cache_path=os.path.join(args.cache_dir, "output", ORGANIZATION),
            key_value_cache_config=cache_config,
        )
        deep_floyd.run_inference(args.requests_path)

    hlog("Done.")
