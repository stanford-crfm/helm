from tqdm import tqdm
import os
import shutil

from cleanfid import fid

from helm.common.general import ensure_directory_exists, generate_unique_id, get_file_name, safe_symlink
from helm.common.gpu_utils import get_torch_device
from helm.common.request import RequestResult
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService


class FIDMetric(Metric):
    """
    Frechet Inception Distance (FID) is a measure of similarity between two sets of images.
    We used the implementation from "On Aliased Resizing and Surprising Subtleties in GAN Evaluation"
    (https://arxiv.org/abs/2104.11222).

    Note: The set of images has to be large (tens of thousands of images) to get reasonable FID scores.

    @misc{https://doi.org/10.48550/arxiv.2104.11222,
        doi = {10.48550/ARXIV.2104.11222},
        url = {https://arxiv.org/abs/2104.11222},
        author = {Parmar, Gaurav and Zhang, Richard and Zhu, Jun-Yan},
        keywords = {Computer Vision and Pattern Recognition (cs.CV), Graphics (cs.GR), Machine Learning (cs.LG),
        FOS: Computer and information sciences, FOS: Computer and information sciences},
        title = {On Aliased Resizing and Surprising Subtleties in GAN Evaluation},
        publisher = {arXiv},
        year = {2021},
        copyright = {arXiv.org perpetual, non-exclusive license}
    }

    TODO: delete FIDMetric. Use FidelityMetric instead.
    """

    def __repr__(self):
        return "FIDMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        # pytorch_fid requires the two sets of images to be in two separate directories.
        # Gather the images by relying on symlinks.
        generated_images_path: str = os.path.join(eval_cache_path, generate_unique_id())
        ensure_directory_exists(generated_images_path)
        gold_images_path: str = os.path.join(eval_cache_path, generate_unique_id())
        ensure_directory_exists(gold_images_path)

        for request_state in tqdm(scenario_state.request_states):
            assert request_state.result is not None
            request_result: RequestResult = request_state.result

            # Gather the model-generated images
            # TODO: use CLIP to pick the best generated image when `num_completions` > 1
            for image in request_result.completions:
                assert image.file_location is not None
                safe_symlink(
                    src=image.file_location,
                    dest=os.path.join(generated_images_path, get_file_name(image.file_location)),
                )

            # Gather the gold images
            instance: Instance = request_state.instance
            for reference in instance.references:
                if not reference.is_correct:
                    continue

                assert reference.output.file_path is not None
                file_path: str = reference.output.file_path
                safe_symlink(src=file_path, dest=os.path.join(gold_images_path, get_file_name(file_path)))

        # Calculate the FID score between model-generated and gold images
        fid_score: float = fid.compute_fid(
            generated_images_path,
            gold_images_path,
            device=get_torch_device(),
            num_workers=0,  # Have to set to 0 (see https://github.com/GaParmar/clean-fid/issues/17)
        )

        # Clean up the symlinks and delete the temp directories
        shutil.rmtree(generated_images_path)
        shutil.rmtree(gold_images_path)

        return MetricResult(aggregated_stats=[Stat(MetricName("fid_alternate")).add(fid_score)], per_instance_stats=[])
