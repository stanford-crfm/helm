from tqdm import tqdm
from typing import Dict, List, Set, Optional
import math
import os
import shutil

import torch
import torch_fidelity

from helm.common.general import copy_image, ensure_directory_exists, generate_unique_id, get_file_name, hlog
from helm.common.request import RequestResult
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from .images_utils import is_blacked_out_image


class FidelityMetric(Metric):
    """
    Frechet Inception Distance (FID) is a measure of similarity between two sets of images.
    Inception Score (IS) measures quality and diversity of images.
    Both metrics require a large number of samples to compute.

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

    # The Stable Diffusion paper (https://arxiv.org/abs/2112.10752) computed FID with 256x256 images
    IMAGE_WIDTH: int = 256
    IMAGE_HEIGHT: int = 256

    def __repr__(self):
        return "FidelityMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        dest_path: str
        unique_perturbations: Set[Optional[PerturbationDescription]] = set()

        gold_images_path: str = os.path.join(eval_cache_path, generate_unique_id())
        ensure_directory_exists(gold_images_path)

        # The library requires the gold and generated images to be in two separate directories.
        # Gather the gold images and the unique perturbations
        for request_state in tqdm(scenario_state.request_states):
            instance: Instance = request_state.instance
            unique_perturbations.add(instance.perturbation)

            for reference in instance.references:
                if not reference.is_correct:
                    continue

                assert reference.output.file_path is not None
                file_path: str = reference.output.file_path
                dest_path = os.path.join(gold_images_path, get_file_name(file_path))
                copy_image(file_path, dest_path, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT)

        # Compute the FID for each perturbation group
        stats: List[Stat] = []
        for perturbation in unique_perturbations:
            perturbation_name: str = "" if perturbation is None else str(perturbation)
            generated_images_path: str = os.path.join(eval_cache_path, generate_unique_id())
            ensure_directory_exists(generated_images_path)

            for request_state in tqdm(scenario_state.request_states):
                if request_state.instance.perturbation != perturbation:
                    continue

                assert request_state.result is not None
                request_result: RequestResult = request_state.result

                # Gather the model-generated images
                for image in request_result.completions:
                    if image.file_path is not None and not is_blacked_out_image(image.file_path):
                        dest_path = os.path.join(generated_images_path, get_file_name(image.file_path))
                        copy_image(image.file_path, dest_path, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT)

            # The torch_fidelity library fails when there are too few images (i.e., `max_eval_instances` is small).
            try:
                metrics_dict: Dict[str, float] = torch_fidelity.calculate_metrics(
                    input1=generated_images_path,
                    input2=gold_images_path,
                    isc=True,
                    fid=True,
                    cuda=torch.cuda.is_available(),
                    save_cpu_ram=not torch.cuda.is_available(),
                )
                fid: float = metrics_dict["frechet_inception_distance"]
                inception_score: float = metrics_dict["inception_score_mean"]
                stats.extend(
                    [
                        Stat(MetricName("FID" + f"_{perturbation_name}" if perturbation_name else "")).add(fid),
                        Stat(MetricName("inception_score" + f"_{perturbation_name}" if perturbation_name else "")).add(
                            inception_score if not math.isnan(inception_score) else 0
                        ),
                    ]
                )
            except AssertionError as e:
                hlog(f"Error occurred when computing fidelity metrics for perturbation: {perturbation_name} Error: {e}")

            shutil.rmtree(generated_images_path)

        # Delete the gold images directory
        shutil.rmtree(gold_images_path)

        return MetricResult(aggregated_stats=stats, per_instance_stats=[])
