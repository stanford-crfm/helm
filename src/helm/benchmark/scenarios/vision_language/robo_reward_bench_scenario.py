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
        policy_label: str,
        annotation: dict,
        output_path: str,
        insert_newline_before: bool = False,
        omit_policy_label: bool = False,
        num_videos: Optional[int] = None,
    ) -> List[MediaObject]:
        content = []
        if insert_newline_before:
            content.append(MediaObject(text="\n", content_type="text/plain"))

        policy_title: str = f"Policy {policy_label}"
        if not omit_policy_label:
            content.append(MediaObject(text=policy_title, content_type="text/plain"))

        policy_key: str = f"policy_{policy_label.lower()}"
        policy_rollout: dict = annotation[policy_key]

        # possibly limit how many cameras we include
        cameras = policy_rollout["cameras"]
        if num_videos is not None:
            cameras = cameras[:num_videos]

        for camera in cameras:
            cam_path: str = RoboRewardBenchPreferenceRankingScenario.get_full_cam_path(
                output_path, camera["camera_path"]
            )
            cam_name: str = camera["name"]
            if cam_path is not None:
                camera_header: str = f"{cam_name.capitalize()} camera:"
                if not omit_policy_label:
                    camera_header = f"{policy_title} - {camera_header}"
                content.extend(
                    [
                        MediaObject(text=camera_header, content_type="text/plain"),
                        MediaObject(location=cam_path, content_type="video/mp4"),
                    ]
                )

        return content

    def get_instances(self, output_path: str) -> List[Instance]:
        target_path: str = os.path.join(output_path, "output")
        annotation_path: str = os.path.join(target_path, "robo_arena_2025-06-21.jsonl")
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

            content.extend(
                self.add_policy_videos(policy_label="A", annotation=annotation, output_path=output_path)
            )
            content.extend(
                self.add_policy_videos(
                    policy_label="B",
                    annotation=annotation,
                    output_path=output_path,
                    insert_newline_before=True,
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


class RoboRewardBenchProgressPredictionScenario(RoboRewardBenchPreferenceRankingScenario):
    def get_instances(self, output_path: str) -> List[Instance]:
        annotation_path: str = os.path.join(output_path, "output", "robo_arena_2025-06-21.jsonl")
        assert os.path.exists(annotation_path), f"Annotation file does not exist at path: {annotation_path}"

        with open(annotation_path, "r") as f:
            head_to_head_annotations = [json.loads(line) for line in f.readlines()]

        instances: List[Instance] = []
        for annotation in head_to_head_annotations:
            annotation_id: str = annotation["id"]
            prompt: str = f"Task: {annotation['task']}"

            for policy_label in ["A", "B"]:
                content: List[MediaObject] = [
                    MediaObject(text=prompt + "\n", content_type="text/plain"),
                ]
                content.extend(
                    self.add_policy_videos(
                        policy_label=policy_label,
                        annotation=annotation,
                        output_path=output_path,
                        omit_policy_label=True,
                    )
                )

                policy_key: str = f"policy_{policy_label.lower()}"
                policy_rollout: dict = annotation[policy_key]
                # Assuming the partial success score is a float between 0 and 1
                partial_success_score: float = policy_rollout["partial_success_score"] * 100
                instances.append(
                    Instance(
                        id=f"{annotation_id}_{policy_label}",
                        input=Input(multimedia_content=MultimediaObject(content)),
                        references=[Reference(output=Output(text=str(partial_success_score)), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                    )
                )

        return instances


class RoboRewardBenchScenario(Scenario):
    name = "robo_reward_bench"
    description = "Rate robot task completion on a discrete 1-5 scale from a single video."
    tags = ["vision-language", "video"]

    output_path: str = "/nlp/scr4/nlp/crfm/text2image/text2image-rlhf/robotics/roboreward/roboreward"

    def __init__(self, subset: str) -> None:
        super().__init__()
        self._subset: str = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        output_path = os.path.join(self.output_path, "collected")
        annotation_path: str = os.path.join(output_path, "final_test.json")
        assert os.path.exists(annotation_path), f"Annotation file does not exist at path: {annotation_path}"

        with open(annotation_path, "r") as f:
            examples = json.load(f)

        instances: List[Instance] = []
        for idx, ex in enumerate(examples):
            subset: str = ex["subset"]
            if subset != self._subset:
                continue

            conversations: dict = ex["conversations"]
            user_turn: dict = conversations[0]
            assert user_turn["from"] == "human"
            prompt = user_turn["value"].replace("<video>", "")

            video_path: str = os.path.join(output_path, ex["video"][0])
            assert os.path.exists(video_path), f"Video path does not exist: {video_path}"

            reward: int = ex["metadata"]["reward"]
            assert 1 <= reward <= 5, f"Invalid reward: {reward}"
            gpt_response: str = conversations[1]["value"]
            assert str(reward) in gpt_response, f"Rewards do not match: {reward} vs {gpt_response}"

            content: List[MediaObject] = [
                MediaObject(text=prompt, content_type="text/plain"),
                MediaObject(location=video_path, content_type="video/mp4"),
            ]

            original_id: str = ex["metadata"]["original_id"]
            instances.append(
                Instance(
                    id=f"{original_id}:{idx}",
                    input=Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(output=Output(text=str(reward)), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )
        return instances
