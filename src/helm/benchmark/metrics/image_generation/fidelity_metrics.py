from tqdm import tqdm
from typing import Dict, List, Set, Optional
import math
import os
import shutil

from helm.common.general import ensure_directory_exists, generate_unique_id, get_file_name, hlog
from helm.common.gpu_utils import is_cuda_available, get_torch_device
from helm.common.request import RequestResult
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import MetricInterface, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.images_utils import is_blacked_out_image, copy_image
from helm.common.optional_dependencies import handle_module_not_found_error


class FidelityMetric(MetricInterface):
    """
    Frechet Inception Distance (FID) is a measure of similarity between two sets of images.
    Inception Score (IS) measures quality and diversity of images.
    Both metrics require a large number of samples to compute.

    @misc{Seitzer2020FID,
      author={Maximilian Seitzer},
      title={{pytorch-fid: FID Score for PyTorch}},
      month={August},
      year={2020},
      note={Version 0.3.0},
      howpublished={https://github.com/mseitzer/pytorch-fid},
    }

    @misc{obukhov2020torchfidelity,
      author={Anton Obukhov and Maximilian Seitzer and Po-Wei Wu and Semen Zhydenko and Jonathan Kyl
              and Elvis Yu-Jing Lin},
      year=2020,
      title={High-fidelity performance metrics for generative models in PyTorch},
      url={https://github.com/toshas/torch-fidelity},
      publisher={Zenodo},
      version={v0.3.0},
      doi={10.5281/zenodo.4957738},
      note={Version: 0.3.0, DOI: 10.5281/zenodo.4957738}
    }
    """

    IMAGE_WIDTH: int = 512
    IMAGE_HEIGHT: int = 512

    def __repr__(self):
        return "FidelityMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        try:
            import torch_fidelity
            from pytorch_fid.fid_score import calculate_fid_given_paths
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        dest_path: str
        unique_perturbations: Set[Optional[PerturbationDescription]] = set()

        gold_images_path: str = os.path.join(eval_cache_path, generate_unique_id())
        ensure_directory_exists(gold_images_path)

        # The library requires the gold and generated images to be in two separate directories.
        # Gather the gold images and the unique perturbations
        num_gold_images: int = 0
        for request_state in tqdm(scenario_state.request_states):
            instance: Instance = request_state.instance
            unique_perturbations.add(instance.perturbation)

            for reference in instance.references:
                if not reference.is_correct:
                    continue

                assert (
                    reference.output.multimedia_content is not None
                    and reference.output.multimedia_content.media_objects[0].location is not None
                )
                file_path: str = reference.output.multimedia_content.media_objects[0].location
                dest_path = os.path.join(gold_images_path, get_file_name(file_path))
                copy_image(file_path, dest_path, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT)
                num_gold_images += 1
        hlog(f"Resized {num_gold_images} gold images to {self.IMAGE_WIDTH}x{self.IMAGE_HEIGHT}.")

        # Compute the FID for each perturbation group
        stats: List[Stat] = []
        for perturbation in unique_perturbations:
            perturbation_name: str = "" if perturbation is None else str(perturbation)
            generated_images_path: str = os.path.join(eval_cache_path, generate_unique_id())
            ensure_directory_exists(generated_images_path)

            num_generated_images: int = 0
            for request_state in tqdm(scenario_state.request_states):
                if request_state.instance.perturbation != perturbation:
                    continue

                assert request_state.result is not None
                request_result: RequestResult = request_state.result

                # Gather the model-generated images
                for image in request_result.completions:
                    assert image.multimodal_content is not None
                    location = image.multimodal_content.media_objects[0].location
                    if location is not None and not is_blacked_out_image(location):
                        dest_path = os.path.join(generated_images_path, get_file_name(location))
                        copy_image(location, dest_path, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT)
                        num_generated_images += 1

            compute_kid: bool = num_generated_images >= 1000
            hlog(f"Resized {num_generated_images} images to {self.IMAGE_WIDTH}x{self.IMAGE_HEIGHT}.")

            try:
                hlog(f"Computing FID between {generated_images_path} and {gold_images_path}...")
                fid: float = calculate_fid_given_paths(
                    paths=[generated_images_path, gold_images_path],
                    device=get_torch_device(),
                    # Following defaults set in
                    # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L54
                    batch_size=50,
                    dims=2048,
                    num_workers=8,
                )
                hlog(f"Done. FID score: {fid}")

                # The torch_fidelity library fails when there are too few images (i.e., `max_eval_instances` is small).
                hlog("Computing the other fidelity metrics...")
                metrics_dict: Dict[str, float] = torch_fidelity.calculate_metrics(
                    input1=generated_images_path,
                    input2=gold_images_path,
                    isc=True,
                    fid=False,
                    kid=compute_kid,
                    ppl=False,  # Requires `GenerativeModel`
                    cuda=is_cuda_available(),
                    save_cpu_ram=not is_cuda_available(),
                )
                inception_score: float = metrics_dict["inception_score_mean"]
                if math.isnan(inception_score):
                    inception_score = 0

                stats.extend(
                    [
                        Stat(MetricName("fid", perturbation=perturbation)).add(fid),
                        Stat(MetricName("inception_score", perturbation=perturbation)).add(inception_score),
                    ]
                )
                if compute_kid:
                    kid: float = metrics_dict["kernel_inception_distance_mean"]
                    stats.append(Stat(MetricName("kernel_inception_distance", perturbation=perturbation)).add(kid))
            except AssertionError as e:
                hlog(f"Error occurred when computing fidelity metrics for perturbation: {perturbation_name} Error: {e}")

            shutil.rmtree(generated_images_path)

        # Delete the gold images directory
        shutil.rmtree(gold_images_path)

        return MetricResult(aggregated_stats=stats, per_instance_stats=[])
