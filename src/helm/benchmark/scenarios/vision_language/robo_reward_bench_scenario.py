from typing import List, Optional
import json
import os

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject


class RoboRewardBenchScenario(Scenario):
    name = "robo_reward_bench"
    description = "To evaluate how well VLMs can judge rollouts of robot actions in videos."
    tags = ["vision-language", "video"]

    def get_instances(self, output_path: str) -> List[Instance]:
        def get_full_cam_path(cam_path: str) -> Optional[str]:
            if cam_path is None:
                return None
            full_path: str = os.path.join(output_path, cam_path)
            assert os.path.exists(full_path), f"Camera path does not exist: {full_path}"
            return full_path

        target_path: str = os.path.join(output_path, "output")
        annotation_path: str = os.path.join(target_path, "robo_arena.jsonl")
        assert os.path.exists(annotation_path), f"Annotation file does not exist at path: {annotation_path}"

        with open(annotation_path, "r") as f:
            head_to_head_annotations = [json.loads(line) for line in f.readlines()]

        instances: List[Instance] = []
        for annotation in head_to_head_annotations:
            annotation_id: str = annotation["id"]
            prompt: str = f"Task: {annotation['prompt']}"
            winner: Optional[str] = annotation["won"]  # "A", "B", or "tie"

            content: List[MediaObject] = [
                MediaObject(text=prompt + "\n", content_type="text/plain"),
            ]

            def add_policy_videos(policy_label: str, insert_newline_before: bool = False):
                if insert_newline_before:
                    content.append(MediaObject(text="\n", content_type="text/plain"))

                policy_title: str = f"Policy {policy_label}"
                content.append(MediaObject(text=policy_title, content_type="text/plain"))
                for cam_type in ["left", "right", "wrist"]:
                    key = f"policy_{policy_label.lower()}_{cam_type}_cam_path"
                    cam_path = get_full_cam_path(annotation.get(key))
                    if cam_path is not None:
                        content.extend([
                            MediaObject(
                                text=f"{policy_title} - {cam_type.capitalize()} camera:", content_type="text/plain"
                            ),
                            MediaObject(location=cam_path, content_type="video/mp4"),
                        ])

            add_policy_videos("A")
            add_policy_videos("B", insert_newline_before=True)

            correct_output: str = winner if winner in ["A", "B", "tie"] else "tie"

            instances.append(
                Instance(
                    id=annotation_id,
                    input=Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(output=Output(text=correct_output), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
