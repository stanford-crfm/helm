from collections import defaultdict
from typing import List
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
from helm.common.general import ensure_file_downloaded


class MSRVTTScenario(Scenario):
    """
    A large-scale video benchmark for video understanding, especially the emerging task of translating video to text.
    This is achieved by collecting 257 popular queries from a commercial video search engine, with 118 videos for
    each query. In its current version, MSR-VTT provides 10K web video clips with 41.2 hours and 200K clip-sentence
    pairs in total, covering the most comprehensive categories and diverse visual content, and representing the
    largest dataset in terms of sentence and vocabulary. Each clip is annotated with about 20 natural sentences
    by 1,327 AMT workers.

    Website link: https://cove.thecvf.com/datasets/839

    Citation:
    MSR-VTT: A Large Video Description Dataset for Bridging Video and Language Jun Xu, Tao Mei, Ting Yao, Yong Rui
    CVPR 2016
    """

    DOWNLOAD_URL: str = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"

    name = "msr_vtt"
    description = "Video captioning dataset with 10K web video clips and 200K clip-sentence pairs."
    tags = ["vision-language", "video", "captioning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the dataset
        target_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url=self.DOWNLOAD_URL,
            target_path=target_path,
            unpack=True,
        )

        annotation_path: str = os.path.join(target_path, "annotation", "MSR_VTT.json")
        with open(annotation_path, "r") as f:
            annotations = json.load(f)["annotations"]

        video_path_to_annotations: dict[str, set[str]] = defaultdict(set)
        for annotation in annotations:
            video_id: str = annotation["image_id"]
            video_path: str = os.path.join(target_path, "videos", "all", f"{video_id}.mp4")
            assert os.path.exists(video_path), f"Video does not exist at path: {video_path}"
            video_path_to_annotations[video_path].add(annotation["caption"])

        instances: List[Instance] = []
        for video_path, captions in video_path_to_annotations.items():
            content: List[MediaObject] = [
                MediaObject(location=video_path, content_type="video/mp4"),
            ]
            references: List[Reference] = [Reference(Output(text=caption), tags=[CORRECT_TAG]) for caption in captions]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
