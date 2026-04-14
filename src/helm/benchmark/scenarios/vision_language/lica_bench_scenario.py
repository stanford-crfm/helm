"""
LICA-Bench: evaluation of vision-language models on graphic design understanding and generation.

This scenario wraps tasks from `lica-bench` (import name ``design_benchmarks``): layout, typography,
SVG, templates, animation, and related benchmarks over the
`Lica dataset <https://github.com/purvanshi/lica-dataset>`_.

Install the optional HELM extra ``crfm-helm[lica-bench]``, download the dataset bundle
(``lica-benchmarks-dataset/``), and set ``dataset_root`` on the run entry or the environment
variable ``LICA_BENCH_DATASET_ROOT``.

References:

- Benchmark code: https://github.com/purvanshi/lica-bench
- Paper-style description: see the lica-bench README on GitHub

**Metrics:** HELM reports standard text generation metrics (exact match, ROUGE, etc.). Task-specific
scores from lica-bench (for example top-5 accuracy on category tasks) should be computed with the
native ``lica-bench`` runner when needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Instance, Input, Output, Reference, Scenario
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from helm.common.media_object import MediaObject, MultimediaObject

LICA_BENCH_DATASET_ROOT_ENV: str = "LICA_BENCH_DATASET_ROOT"

# Cap appended JSON metadata so prompts stay bounded.
_MAX_METADATA_CHARS: int = 200_000


def _require_lica_bench():
    try:
        from design_benchmarks import BenchmarkRegistry  # noqa: F401
        from design_benchmarks.models.base import ModelInput  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The lica-bench scenario requires the optional dependency. Install with:\n"
            '  pip install "crfm-helm[lica-bench]"'
        ) from exc


def _extension_to_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".svg": "image/svg+xml",
    }.get(ext, "image/png")


def _serialize_ground_truth(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _write_bytes_image(data: bytes, tmp_dir: str, stem: str) -> str:
    ensure_directory_exists(tmp_dir)
    path = os.path.join(tmp_dir, f"{stem}.bin.png")
    with open(path, "wb") as f:
        f.write(data)
    return path


class LicaBenchScenario(Scenario):
    """Scenario backed by a single lica-bench task (e.g. ``category-1``, ``svg-1``)."""

    name: str = "lica_bench"
    description: str = (
        "Graphic design benchmarks from "
        "[lica-bench](https://github.com/purvanshi/lica-bench) "
        "(layout, typography, SVG, templates, temporal, Lottie, category)."
    )
    tags: List[str] = ["vision-language", "lica-bench"]

    def __init__(self, benchmark_id: str, dataset_root: str = "", max_instances: Optional[int] = None):
        super().__init__()
        self._benchmark_id: str = benchmark_id
        resolved_root = (dataset_root or "").strip() or os.environ.get(LICA_BENCH_DATASET_ROOT_ENV, "")
        if not resolved_root:
            raise ValueError(
                "LicaBenchScenario needs a dataset root: pass dataset_root in the scenario args, "
                f"or set the {LICA_BENCH_DATASET_ROOT_ENV} environment variable to your "
                "lica-benchmarks-dataset directory."
            )
        self._dataset_root: str = str(Path(resolved_root).expanduser().resolve())
        self._max_instances: Optional[int] = int(max_instances) if max_instances is not None else None

    def _model_input_to_multimedia(self, model_input: Any, sample_id: str, tmp_dir: str) -> Optional[MultimediaObject]:
        from design_benchmarks.models.base import ModelInput

        if not isinstance(model_input, ModelInput):
            hlog(f"Skipping sample {sample_id}: expected ModelInput, got {type(model_input)!r}.")
            return None

        text = model_input.text or ""
        meta = model_input.metadata or {}
        if meta:
            meta_str = json.dumps(meta, ensure_ascii=False, default=str)
            if len(meta_str) > _MAX_METADATA_CHARS:
                meta_str = meta_str[:_MAX_METADATA_CHARS] + "\n...[truncated]"
            text = f"{text}\n\n[benchmark_metadata JSON]\n{meta_str}" if text else f"[benchmark_metadata JSON]\n{meta_str}"

        media_objects: List[MediaObject] = []
        if text:
            media_objects.append(MediaObject(text=text, content_type="text/plain"))

        any_media_ok = False
        for i, img in enumerate(model_input.images or []):
            location: Optional[str] = None
            if isinstance(img, (str, Path)):
                location = str(Path(img).expanduser().resolve())
                if not os.path.isfile(location):
                    hlog(f"Skipping sample {sample_id}: missing media file: {location}")
                    return None
            elif isinstance(img, bytes):
                location = _write_bytes_image(img, tmp_dir, f"{sample_id}_{i}")
            else:
                hlog(f"Skipping sample {sample_id}: unsupported image payload type {type(img)!r}.")
                return None

            media_objects.append(MediaObject(location=location, content_type=_extension_to_mime(location)))
            any_media_ok = True

        if not media_objects:
            hlog(f"Skipping sample {sample_id}: empty model input.")
            return None

        # Multimodal adapter expects non-empty multimedia_content; text-only is valid.
        if not any_media_ok and not text:
            return None

        return MultimediaObject(media_objects=media_objects)

    def get_instances(self, output_path: str) -> List[Instance]:
        _require_lica_bench()
        from design_benchmarks import BenchmarkRegistry
        from design_benchmarks.models.base import Modality

        registry = BenchmarkRegistry()
        registry.discover()
        bench = registry.get(self._benchmark_id)

        data_dir = bench.resolve_data_dir(self._dataset_root)
        samples = bench.load_data(data_dir, n=self._max_instances, dataset_root=self._dataset_root)

        tmp_dir = os.path.join(output_path, "lica_bench_media")
        ensure_directory_exists(tmp_dir)

        instances: List[Instance] = []
        for sample in samples:
            sid = str(sample.get("sample_id", f"idx_{len(instances)}"))
            model_input = bench.build_model_input(sample, modality=Modality.TEXT_AND_IMAGE)
            multimedia = self._model_input_to_multimedia(model_input, sid, tmp_dir)
            if multimedia is None:
                continue

            ref_text = _serialize_ground_truth(sample.get("ground_truth", ""))
            instances.append(
                Instance(
                    input=Input(text="", multimedia_content=multimedia),
                    references=[Reference(output=Output(text=ref_text), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                    id=sid,
                )
            )
        return instances
