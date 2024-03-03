from collections import Counter
from dacite import from_dict
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
import json
import os
import time

from diffusers import DiffusionPipeline
import torch

from helm.common.cache import (
    KeyValueStore,
    KeyValueStoreCacheConfig,
    MongoCacheConfig,
    SqliteCacheConfig,
    create_key_value_store,
)
from helm.common.request import Request
from helm.common.file_caches.local_file_cache import LocalFileCache
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.clients.image_generation.deep_floyd_client import DeepFloydClient


"""
Script to run inference for the DeepFloyd-IF models given a dry run benchmark output folder of requests.

From https://huggingface.co/docs/diffusers/main/en/api/pipelines/if#text-to-image-generation

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

The following dependencies need to be installed in order to run inference with DeepFloyd models:

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

Example usage (after a dryrun with run suite deepfloyd):

python3 scripts/offline_eval/deepfloyd/deepfloyd.py IF-I-XL-v1.0 benchmark_output/runs/deepfloyd \
--mongo-uri <MongoDB address>

"""

ORGANIZATION: str = "DeepFloyd"


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

    def _process_request(self, request_state: Dict, store: KeyValueStore) -> bool:
        request: Request = from_dict(Request, request_state["request"])
        raw_request: Dict = DeepFloydClient.convert_to_raw_request(request)

        if store.contains(raw_request):
            return True

        image_paths: List[str] = []
        start_time: float = time.time()
        for i in range(request.num_completions):
            file_path: str = self._file_cache.generate_unique_new_file_path()
            self._run_inference_single_image(request.prompt, file_path, i)
            image_paths.append(file_path)
        total_inference_time: float = time.time() - start_time

        result: Dict = {"images": image_paths, "total_inference_time": total_inference_time}
        store.put(raw_request, result)
        return False

    def run_all(self, run_suite_path: str):
        """
        Given a run suite folder, runs inference for all the requests.
        """

        counts = Counter(inference_count=0, cached_count=0)

        # Go through all the valid run folders, pull requests from the scenario_state.json
        # files and run inference for each request.
        with create_key_value_store(self._key_value_cache_config) as store:
            for run_dir in tqdm(os.listdir(run_suite_path)):
                run_path: str = os.path.join(run_suite_path, run_dir)

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

                        if current_model_engine != self._model_engine:
                            hlog(f"Not running inference for {current_model_engine}.")
                            continue

                        for request_state in tqdm(scenario_state["request_states"]):
                            cached: bool = self._process_request(request_state, store)
                            counts["cached_count" if cached else "inference_count"] += 1

        hlog(
            f"Processed {counts['inference_count']} requests. "
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
    parser.add_argument("model_name", type=str, help="Name of the model", choices=DeepFloyd.MODEL_NAME_TO_MODELS.keys())
    parser.add_argument("run_suite_path", type=str, help="Path to run path.")
    args = parser.parse_args()

    cache_config: KeyValueStoreCacheConfig
    if args.mongo_uri:
        hlog(f"Initialized MongoDB cache with URI: {args.mongo_uri}")
        cache_config = MongoCacheConfig(args.mongo_uri, ORGANIZATION)
    elif args.cache_dir:
        hlog(f"WARNING: Initialized SQLite cache at path: {args.cache_dir}. Are you debugging??")
        cache_config = SqliteCacheConfig(os.path.join(args.cache_dir, f"{ORGANIZATION}.sqlite"))
    else:
        raise ValueError("Either --cache-dir or --mongo-uri should be specified")

    deep_floyd = DeepFloyd(
        model_name=args.model_name,
        file_cache_path=os.path.join(args.cache_dir, "output", ORGANIZATION),
        key_value_cache_config=cache_config,
    )
    deep_floyd.run_all(args.run_suite_path)
    hlog("Done.")
