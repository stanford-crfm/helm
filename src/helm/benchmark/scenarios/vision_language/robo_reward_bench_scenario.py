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


class RoboRewardBenchPreferenceRankingScenario(Scenario):
    name = "robo_reward_bench"
    description = "To evaluate how well VLMs can judge rollouts of robot actions in videos."
    tags = ["vision-language", "video"]

    @staticmethod
    def get_full_cam_path(output_path: str, cam_path: str) -> Optional[str]:
        if cam_path is None:
            return None

        full_path: str = os.path.join(output_path, cam_path)
        assert os.path.exists(full_path), f"Camera path does not exist: {full_path}"
        return full_path

    @staticmethod
    def add_policy_videos(
        policy_label: str, annotation: dict, output_path: str, insert_newline_before: bool = False
    ) -> List[MediaObject]:
        content = []
        if insert_newline_before:
            content.append(MediaObject(text="\n", content_type="text/plain"))

        policy_title: str = f"Policy {policy_label}"
        content.append(MediaObject(text=policy_title, content_type="text/plain"))

        policy_key: str = f"policy_{policy_label.lower()}"
        policy_rollout: dict = annotation[policy_key]

        for camera in policy_rollout["cameras"]:
            cam_path: str = RoboRewardBenchPreferenceRankingScenario.get_full_cam_path(
                output_path, camera["camera_path"]
            )
            cam_name: str = camera["name"]
            if cam_path is not None:
                content.extend(
                    [
                        MediaObject(
                            text=f"{policy_title} - {cam_name.capitalize()} camera:", content_type="text/plain"
                        ),
                        MediaObject(location=cam_path, content_type="video/mp4"),
                    ]
                )

        return content

    def get_instances(self, output_path: str) -> List[Instance]:
        target_path: str = os.path.join(output_path, "output")
        annotation_path: str = os.path.join(target_path, "robo_arena.jsonl")
        assert os.path.exists(annotation_path), f"Annotation file does not exist at path: {annotation_path}"

        with open(annotation_path, "r") as f:
            head_to_head_annotations = [json.loads(line) for line in f.readlines()]

        instances: List[Instance] = []
        for annotation in head_to_head_annotations:
            annotation_id: str = annotation["id"]
            prompt: str = f"Task: {annotation['task']}"
            winner: str = annotation["won"]  # "A", "B", or "tie"

            content: List[MediaObject] = [
                MediaObject(text=prompt + "\n", content_type="text/plain"),
            ]

            content.extend(self.add_policy_videos(policy_label="A", annotation=annotation, output_path=output_path))
            content.extend(
                self.add_policy_videos(
                    policy_label="B", annotation=annotation, output_path=output_path, insert_newline_before=True
                )
            )

            instances.append(
                Instance(
                    id=annotation_id,
                    input=Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(output=Output(text=winner), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
