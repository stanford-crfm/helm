import logging
from typing import List

import comet
from torch import nn

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog
from helm.common.request import RequestResult


class CometMetric(Metric):
    """COMET machine translation metric using a regression model.
    The model takes a triplet of source sentence, translation, and reference
    and computes a score in the range [0, 1] reflecting the quality of the predicted
    translation.

    Paper:
    @inproceedings{rei-etal-2022-comet,
        title = "{COMET}-22: Unbabel-{IST} 2022 Submission for the Metrics Shared Task",
        author = "Rei, Ricardo  and
            C. de Souza, Jos{\'e} G.  and
            Alves, Duarte  and
            Zerva, Chrysoula  and
            Farinha, Ana C  and
            Glushkova, Taisiya  and
            Lavie, Alon  and
            Coheur, Luisa  and
            Martins, Andr{\'e} F. T.",
            editor = {Koehn, Philipp  and
            Barrault, Lo{\"\i}c  and
            Bojar, Ond{\v{r}}ej  and
            Bougares, Fethi  and
            Chatterjee, Rajen  and
            Costa-juss{\`a}, Marta R.  and
            Federmann, Christian  and
            Fishel, Mark  and
            Fraser, Alexander  and
            Freitag, Markus  and
            Graham, Yvette  and
            Grundkiewicz, Roman  and
            Guzman, Paco  and
            Haddow, Barry  and
            Huck, Matthias  and
            Jimeno Yepes, Antonio  and
            Kocmi, Tom  and
            Martins, Andr{\'e}  and
            Morishita, Makoto  and
            Monz, Christof  and
            Nagata, Masaaki  and
            Nakazawa, Toshiaki  and
            Negri, Matteo  and
            N{\'e}v{\'e}ol, Aur{\'e}lie  and
            Neves, Mariana  and
            Popel, Martin  and
            Turchi, Marco  and
            Zampieri, Marcos},
        booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
        month = dec,
        year = "2022",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.wmt-1.52",
    }
    """

    METRIC_NAME = "comet"

    def __init__(self, task: str, model_name: str = "Unbabel/wmt22-comet-da", device: str = "cpu"):
        self.model_name = model_name
        self.comet_scorer: nn.Module = self._load_model(model_name)
        self.num_gpus = 0 if device == "cpu" else 1

        # suppress warnings from PyTorch Lightning which spams terminal
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

    @staticmethod
    def _load_model(model_name: str) -> nn.Module:
        """Load Comet model from the checkpoint.

        Returns:
            The loaded model.
        """
        return comet.load_from_checkpoint(comet.download_model(model_name))

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        hlog(
            f"Setting parallelism from {parallelism} to 1, since "
            f"evaluating {self.__class__.__name__} with parallelism > 1 seg faults."
        )
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=1)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Compute the COMET score for this instance"""
        assert len(request_state.instance.references) == 1
        ref = request_state.instance.references[0].output.text
        src = request_state.instance.input.text

        result = request_state.result
        if not isinstance(result, RequestResult):
            raise TypeError(f"Expected a valid result, but got {result}!")
        mt = result.completions[0].text.strip()

        # comet requires this exac5 format
        data = [dict(ref=ref, src=src, mt=mt)]
        output = self.comet_scorer.predict(data, gpus=self.num_gpus, progress_bar=False)  # type: ignore
        comet_score = output[0][0]  # extract the actual score

        metric_result = [Stat(MetricName(self.METRIC_NAME)).add(comet_score)]

        return metric_result
