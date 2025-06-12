import os
from typing import Dict, Any

import torch

from helm.benchmark.runner import get_cached_models_path
from helm.common.general import ensure_file_downloaded, hlog
from helm.common.images_utils import open_image
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.gpu_utils import get_torch_device
from helm.benchmark.metrics.image_generation.detectors.base_detector import BaseDetector


MODEL_CONFIG_DOWNLOAD_URL: str = "https://drive.google.com/uc?id=1MLuwQ0ZN0gJQ42oVCc0aFz6Rneb1g3Rt"
MODEL_CHECKPOINT_DOWNLOAD_URL: str = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl"
)


class ViTDetDetector(BaseDetector):
    def __init__(self):
        try:
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import LazyConfig
            from detectron2.config import instantiate
            from detectron2.data.catalog import MetadataCatalog
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        super().__init__()

        cache_path: str = get_cached_models_path()
        cfg_path: str = os.path.join(cache_path, "vitdet_model.yaml")
        ensure_file_downloaded(source_url=MODEL_CONFIG_DOWNLOAD_URL, target_path=cfg_path)
        cfg = LazyConfig.load(cfg_path)

        model_path: str = os.path.join(cache_path, "vitdet_model.pkl")
        ensure_file_downloaded(source_url=MODEL_CHECKPOINT_DOWNLOAD_URL, target_path=model_path)
        cfg.train.init_checkpoint = model_path

        model = instantiate(cfg.model).cuda()
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

        self._cfg = cfg
        self._model = model
        self._device: torch.device = get_torch_device()
        hlog("Initialized the ViTDet model.")

        # COCO classes
        self._coco_classes = MetadataCatalog.get("coco_2017_val").thing_classes

    def forward_model(self, image_location: str) -> float:
        try:
            from detectron2.data.common import DatasetFromList, MapDataset
            from detectron2.config import instantiate
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        image = open_image(image_location)
        dataset_dicts = [
            {
                "file_name": image_location,
                "width": image.width,
                "height": image.height,
            }
        ]
        dataset = DatasetFromList(dataset_dicts, copy=False)
        mapper = instantiate(self._cfg.dataloader.test.mapper)
        dataset = MapDataset(dataset, mapper)
        inputs = [dataset[0]]
        outputs = self._model(inputs)
        return outputs[0]["instances"]

    def compute_score(self, caption: str, image_location: str, references: Dict[str, Any]) -> float:
        # hlog(f'compute score for prompt: {caption}, file: {image_location}, skill: {references["skill"]}')
        instances = self.forward_model(image_location)
        if references["skill"] == "object":
            return self.compute_score_object(instances, references)
        if references["skill"] == "count":
            return self.compute_score_count(instances, references)
        if references["skill"] == "spatial":
            return self.compute_score_spatial(instances, references)
        raise NotImplementedError(references["skill"])

    def compute_score_object(self, instances, references):
        gt_class_name = references["object"]
        gt_class = self._coco_classes.index(gt_class_name)
        if len(instances.scores) == 0:
            pred_id = None
            pred_score = torch.zeros(())
            pred_class = None
            pred_class_name = None
            correct = 0.0
        else:
            pred_id = instances.scores.max(-1).indices
            pred_score = instances.scores[pred_id]  # (num_instances,) -> ()    # noqa
            pred_class = instances.pred_classes[pred_id]  # (num_instances,) -> ()
            pred_class_name = self._coco_classes[pred_class.item()]  # noqa

            correct = float(pred_class == gt_class)

        # hlog(f"pred_class: {pred_class_name}, gt_class: {gt_class_name}, correct: {correct}")
        return correct

    def compute_score_count(self, instances, references):
        # assume that there is only one type of object
        gt_class_name = references["object"]
        gt_class_idx = self._coco_classes.index(gt_class_name)
        gt_count = references["count"]
        if len(instances.scores) == 0:
            pred_count = 0
            correct = 0.0
        else:
            pred_count = (instances.pred_classes == gt_class_idx).sum().item()
            correct = float(pred_count == gt_count)
        return correct

    def compute_score_spatial(self, instances, references):
        gt_class_name_1, gt_class_name_2 = references["objects"]
        gt_class_idx_1 = self._coco_classes.index(gt_class_name_1)
        gt_class_idx_2 = self._coco_classes.index(gt_class_name_2)
        relation = references["relation"].split("_")[0]

        if len(instances.scores) == 0:
            correct = 0
            pred_rel = "no_pred"
        else:
            pred_count_1 = (instances.pred_classes == gt_class_idx_1).sum().item()
            pred_count_2 = (instances.pred_classes == gt_class_idx_2).sum().item()
            if pred_count_1 != 1 or pred_count_2 != 1:
                correct = 0
                pred_rel = "obj_count_mismatch"
            else:
                x11, y11 = instances.pred_boxes[instances.pred_classes == gt_class_idx_1].tensor[0, :2]
                x21, y21 = instances.pred_boxes[instances.pred_classes == gt_class_idx_2].tensor[0, :2]

                x_diff = x11 - x21
                y_diff = y11 - y21

                # FIXME: The code below mimics dall-eval logic. I don't think
                # we need to follow it. Does the case of two objects of same
                # category make sense? Also, I don't know why we need to
                # to ensure something is more "right" than it is "above".
                if gt_class_name_1 == gt_class_name_2:
                    if abs(x_diff) > abs(y_diff):
                        if relation in ["left", "right"]:
                            correct = 1
                            pred_rel = "relation_correct"
                        else:
                            pred_rel = "relation_incorrect"
                            correct = 0
                    else:
                        if relation in ["above", "below"]:
                            pred_rel = "relation_correct"
                            correct = 1
                        else:
                            pred_rel = "relation_incorrect"
                            correct = 0
                else:
                    if abs(x_diff) > abs(y_diff):
                        if x11 < x21:
                            pred_rel = "right"
                        else:
                            pred_rel = "left"
                    else:
                        if y11 > y21:
                            pred_rel = "above"
                        else:
                            pred_rel = "below"

                    if relation == pred_rel:
                        correct = 1
                    else:
                        correct = 0
        return correct
