import shutil
import tempfile
import unittest
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig

from helm.common.media_object import MediaObject, MultimediaObject
from helm.benchmark.scenarios.scenario import Instance, Reference, Input, Output, TEST_SPLIT, TRAIN_SPLIT, CORRECT_TAG
from helm.benchmark.window_services.test_utils import get_tokenizer_service
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION_MULTIMODAL, AdapterFactory
from helm.benchmark.adaptation.adapters.multimodal.in_context_learning_multimodal_adapter import (
    InContextLearningMultimodalAdapter,
)
from helm.benchmark.adaptation.adapters.multimodal.multimodal_prompt import MultimodalPrompt


class TestInContextLearningMultimodalAdapter(unittest.TestCase):
    def setup_method(self, _):
        self._path: str = tempfile.mkdtemp()
        self._tokenizer_service = get_tokenizer_service(self._path, BlackHoleCacheBackendConfig())

    def teardown_method(self, _):
        shutil.rmtree(self._path)

    def test_construct_prompt(self):
        adapter_spec: AdapterSpec = AdapterSpec(
            model="simple/model1",
            model_deployment="simple/model1",
            method=ADAPT_GENERATION_MULTIMODAL,
            global_prefix="[START]",
            instructions="Please answer the following question about the images.",
            input_prefix="Input: ",
            input_suffix="\n",
            output_prefix="Output: ",
            output_suffix="<end_of_utterance>",
            instance_prefix="\n",
            max_train_instances=5,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self._tokenizer_service)
        assert isinstance(adapter, InContextLearningMultimodalAdapter)

        train_instances = [
            Instance(
                Input(
                    multimedia_content=MultimediaObject(
                        [
                            MediaObject(location="http://cat.png", content_type="image/png"),
                            MediaObject(text="Which animal?", content_type="text/plain"),
                        ]
                    )
                ),
                references=[
                    Reference(Output(text="cat"), tags=[CORRECT_TAG]),
                    Reference(Output(text="feline"), tags=[CORRECT_TAG]),
                ],
                split=TRAIN_SPLIT,
            ),
            Instance(
                Input(
                    multimedia_content=MultimediaObject(
                        [
                            MediaObject(location="http://owl.png", content_type="image/png"),
                            MediaObject(text="Which bird?", content_type="text/plain"),
                        ]
                    )
                ),
                references=[Reference(Output(text="owl"), tags=[CORRECT_TAG])],
                split=TRAIN_SPLIT,
            ),
        ]
        eval_instance = Instance(
            Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(location="http://dog.png", content_type="image/png"),
                        MediaObject(text="Which fur animal?", content_type="text/plain"),
                    ]
                )
            ),
            references=[Reference(Output(text="dog"), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
        )
        prompt: MultimodalPrompt = adapter.construct_prompt(
            train_instances, eval_instance, include_output=False, reference_index=None
        )

        self.assertEqual(
            prompt.multimedia_object.text,
            "[START]Please answer the following question about the images."
            "\nInput: Which animal?\nOutput: cat<end_of_utterance>"
            "\nInput: Which bird?\nOutput: owl<end_of_utterance>"
            "\nInput: Which fur animal?\nOutput:",
        )

    def test_construct_prompt_multi_label(self):
        adapter_spec: AdapterSpec = AdapterSpec(
            model="simple/model1",
            model_deployment="simple/model1",
            method=ADAPT_GENERATION_MULTIMODAL,
            global_prefix="[START]",
            instructions="Please answer the following question about the images.",
            input_prefix="Input: ",
            input_suffix="\n",
            output_prefix="Output: ",
            output_suffix="<end_of_utterance>",
            instance_prefix="\n",
            max_train_instances=5,
            multi_label=True,
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self._tokenizer_service)
        assert isinstance(adapter, InContextLearningMultimodalAdapter)

        train_instances = [
            Instance(
                Input(
                    multimedia_content=MultimediaObject(
                        [
                            MediaObject(location="http://cat.png", content_type="image/png"),
                            MediaObject(text="Which animal?", content_type="text/plain"),
                        ]
                    )
                ),
                references=[
                    Reference(Output(text="cat"), tags=[CORRECT_TAG]),
                    Reference(Output(text="feline"), tags=[CORRECT_TAG]),
                    Reference(Output(text="lion"), tags=[]),
                ],
                split=TRAIN_SPLIT,
            ),
            Instance(
                Input(
                    multimedia_content=MultimediaObject(
                        [
                            MediaObject(location="http://owl.png", content_type="image/png"),
                            MediaObject(text="Which bird?", content_type="text/plain"),
                        ]
                    )
                ),
                references=[
                    Reference(Output(text="owl"), tags=[CORRECT_TAG]),
                    Reference(Output(text="night hunting bird"), tags=[CORRECT_TAG]),
                ],
                split=TRAIN_SPLIT,
            ),
        ]
        eval_instance = Instance(
            Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(location="http://dog.png", content_type="image/png"),
                        MediaObject(text="Which fur animal?", content_type="text/plain"),
                    ]
                )
            ),
            references=[
                Reference(Output(text="dog"), tags=[CORRECT_TAG]),
                Reference(Output(text="canine"), tags=[CORRECT_TAG]),
            ],
            split=TEST_SPLIT,
        )
        prompt: MultimodalPrompt = adapter.construct_prompt(
            train_instances, eval_instance, include_output=False, reference_index=None
        )

        self.assertEqual(
            prompt.multimedia_object.text,
            "[START]Please answer the following question about the images."
            "\nInput: Which animal?\nOutput: cat, feline<end_of_utterance>"
            "\nInput: Which bird?\nOutput: owl, night hunting bird<end_of_utterance>"
            "\nInput: Which fur animal?\nOutput:",
        )

    def test_construct_prompt_idefics_instruct_example(self):
        """
        Constructing the same prompt from this example: https://huggingface.co/blog/idefics
        """
        adapter_spec: AdapterSpec = AdapterSpec(
            model="simple/model1",
            model_deployment="simple/model1",
            method=ADAPT_GENERATION_MULTIMODAL,
            input_prefix="User: ",
            input_suffix="<end_of_utterance>",
            output_prefix="\nAssistant: ",
            output_suffix="<end_of_utterance>",
            instance_prefix="\n",
            max_train_instances=1,
            stop_sequences=["<end_of_utterance>"],
        )
        adapter = AdapterFactory.get_adapter(adapter_spec, self._tokenizer_service)
        assert isinstance(adapter, InContextLearningMultimodalAdapter)

        train_instances = [
            Instance(
                Input(
                    multimedia_content=MultimediaObject(
                        [
                            MediaObject(text="What is in this image?", content_type="text/plain"),
                            MediaObject(
                                location="https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                                content_type="image/jpeg",
                            ),
                        ]
                    )
                ),
                references=[
                    Reference(
                        Output(
                            text="This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. "
                            "defix is running on the ground."
                        ),
                        tags=[CORRECT_TAG],
                    )
                ],
                split=TRAIN_SPLIT,
            ),
        ]
        eval_instance = Instance(
            Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(
                            location="https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/"
                            "revision/latest?cb=20110815073052",
                            content_type="image/jpeg",
                        ),
                        MediaObject(text="And who is that?", content_type="text/plain"),
                    ]
                )
            ),
            references=[Reference(Output(text="Julius Caesar"), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
        )
        prompt: MultimodalPrompt = adapter.construct_prompt(
            train_instances, eval_instance, include_output=False, reference_index=None
        )
        self.assertEqual(
            prompt.multimedia_object.text,
            "User: What is in this image?"
            "<end_of_utterance>"
            "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. defix is running "
            "on the ground.<end_of_utterance>"
            "\nUser: "
            "And who is that?<end_of_utterance>\nAssistant:",
        )
