import os
import argparse
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from icetk import icetk as tokenizer
from torchvision.utils import save_image

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.clients.client import Client, CachingClient
from helm.clients.image_generation.cogview2.coglm_strategy import CoglmStrategy
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object


class CogView2Client(Client):
    """
    https://github.com/THUDM/CogView2
    """

    MAX_SEQ_LEN: int = 95
    MODEL_URL: str = "https://nlp.stanford.edu/projects/vhelm/cogview2/sharefs.zip"

    def __init__(self, cache_config: CacheConfig, file_cache: FileCache):
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._args: Optional[argparse.Namespace] = None
        self._strategy: Optional[CoglmStrategy] = None
        self._model = None
        self._srg = None

    def _get_model(self) -> None:
        try:
            from SwissArmyTransformer import get_args
            from helm.clients.image_generation.cogview2.coglm_utils import (
                get_recipe,
                InferenceModel,
            )
            from helm.clients.image_generation.cogview2.sr_pipeline import SRGroup
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        tokenizer.add_special_tokens(["<start_of_image>", "<start_of_english>", "<start_of_chinese>"])

        model_local_path: str = f"{self._file_cache._location}/cogview2"  # type: ignore
        os.environ["SAT_HOME"] = f"{model_local_path}/sharefs/cogview-new"

        # Download the model if not yet
        if not os.path.exists(model_local_path):
            os.system(f"mkdir -p {model_local_path}")
            os.system(f"wget {self.MODEL_URL} -P {model_local_path}")
            os.system(f"unzip {model_local_path}/sharefs.zip -d {model_local_path}")

        if self._model is None:
            # Set up args
            args = get_args("--mode inference --fp16".split())
            self._args = argparse.Namespace(**vars(args), **get_recipe("none"))
            self._args.img_size = 160
            self._args.only_first_stage = False
            self._args.inverse_prompt = False
            self._args.batch_size = 1
            self._args.max_inference_batch_size = 1

            # Load the model components
            self._model, self._args = InferenceModel.from_pretrained(self._args, "coglm")
            invalid_slices = [slice(tokenizer.num_image_tokens, None)]
            self._strategy = CoglmStrategy(
                invalid_slices,
                temperature=getattr(self._args, "temp_all_gen"),
                top_k=getattr(self._args, "topk_gen"),
                top_k_cluster=getattr(self._args, "temp_cluster_gen"),
            )
            self._srg = SRGroup(self._args)  # type: ignore

    def _model_inference(self, prompt) -> torch.Tensor:
        try:
            from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
            from helm.clients.image_generation.cogview2.coglm_utils import get_masks_and_position_ids_coglm
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        with torch.no_grad():
            text = getattr(self._args, "query_template").format(prompt)
            seq = tokenizer.encode(text)
            if len(seq) > self.MAX_SEQ_LEN:
                seq = seq[: self.MAX_SEQ_LEN - 2] + seq[-2:]
            txt_len = len(seq) - 1
            device = getattr(self._args, "device")
            seq = torch.tensor(seq + [-1] * 400, device=device)
            # calibrate text length
            log_attention_weights = torch.zeros(
                len(seq), len(seq), device=device, dtype=torch.half if getattr(self._args, "fp16") else torch.float32
            )
            log_attention_weights[:, :txt_len] = getattr(self._args, "attn_plus")
            # generation
            mbz = getattr(self._args, "max_inference_batch_size")
            batch_size = getattr(self._args, "batch_size")
            assert batch_size < mbz or batch_size % mbz == 0
            get_func = partial(get_masks_and_position_ids_coglm, context_length=txt_len)
            output_list = []
            for tim in range(max(batch_size // mbz, 1)):
                setattr(self._strategy, "start_pos", txt_len + 1)
                coarse_samples = filling_sequence(
                    self._model,
                    seq.clone(),
                    batch_size=min(batch_size, mbz),
                    strategy=self._strategy,
                    log_attention_weights=log_attention_weights,
                    get_masks_and_position_ids=get_func,
                )[0]
                output_list.append(coarse_samples)

            output_tokens = torch.cat(output_list, dim=0)
            images = []
            iter_tokens = getattr(self._srg, "sr_base")(output_tokens[:, -400:], seq[:txt_len])
            for seq in iter_tokens:
                decoded_img = tokenizer.decode(image_ids=seq[-3600:])
                decoded_img = torch.nn.functional.interpolate(decoded_img, size=(480, 480))
                images.append(decoded_img)  # only the last image (target)
            return images[0]

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "prompt": request.prompt,
        }

        try:

            def do_it() -> Dict[str, Any]:
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    self._get_model()

                    images: List[torch.Tensor] = []
                    for _ in range(request.num_completions):
                        output = self._model_inference(**raw_request).cpu()  # (1, 3, 480, 480)
                        images.append(output)

                    assert (
                        len(images) == request.num_completions
                    ), f"Expected {request.num_completions} images, but got {len(images)}"

                    result: Dict = {"file_locations": []}
                    for image in images:
                        # Write out the image to a file and save the path
                        file_location: str = self._file_cache.generate_unique_new_file_path()  # type: ignore
                        save_image(image, file_location, normalize=True)
                        hlog(f"Image saved at {file_location}.")
                        result["file_locations"].append(file_location)
                    return result

            # Include the model name and number of completions in the cache key
            cache_key = CachingClient.make_cache_key(
                {"model": request.model_engine, "n": request.num_completions, **raw_request}, request
            )
            results, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as e:
            error: str = f"CogView2Client error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="", logprob=0, tokens=[], multimodal_content=get_single_image_multimedia_object(location)
            )
            for location in results["file_locations"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=results["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
