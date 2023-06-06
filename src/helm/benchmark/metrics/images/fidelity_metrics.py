from tqdm import tqdm
from typing import Dict, List, Set, Optional
import math
import os
import shutil
import tempfile

import torch
from helm.benchmark.adaptation.request_state import RequestState
import torch_fidelity

from helm.common.general import get_file_name, hlog
from helm.common.request import RequestResult
from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.images_utils import is_blacked_out_image, copy_image


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

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        dest_path: str
        unique_perturbations: Set[Optional[PerturbationDescription]] = set()
        with tempfile.TemporaryDirectory() as gold_images_path, tempfile.TemporaryDirectory() as generated_images_path:
            # The library requires the gold and generated images to be in two separate directories.
            # Gather the gold images and the unique perturbations
            for request_state in tqdm(request_states):
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

                num_generated_images: int = 0
                for request_state in tqdm(request_states):
                    if request_state.instance.perturbation != perturbation:
                        continue

                    assert request_state.result is not None
                    request_result: RequestResult = request_state.result

                    # Gather the model-generated images
                    for image in request_result.completions:
                        if image.file_location is not None and not is_blacked_out_image(image.file_location):
                            dest_path = os.path.join(generated_images_path, get_file_name(image.file_location))
                            copy_image(image.file_location, dest_path, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT)
                            num_generated_images += 1

                compute_kid: bool = num_generated_images >= 1000

                # The torch_fidelity library fails when there are too few images (i.e., `max_eval_instances` is small).
                try:
                    metrics_dict: Dict[str, float] = torch_fidelity.calculate_metrics(
                        input1=generated_images_path,
                        input2=gold_images_path,
                        isc=True,
                        fid=True,
                        kid=compute_kid,
                        ppl=False,  # Requires `GenerativeModel`
                        cuda=torch.cuda.is_available(),
                        save_cpu_ram=not torch.cuda.is_available(),
                    )
                    hlog(f"Computing metrics for perturbation: {perturbation_name if perturbation_name else 'none'}")
                    fid: float = metrics_dict["frechet_inception_distance"]
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

        return stats
