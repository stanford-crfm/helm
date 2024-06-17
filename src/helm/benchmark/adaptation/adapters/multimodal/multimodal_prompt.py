from dataclasses import dataclass
from typing import List

from helm.common.media_object import MediaObject, MultimediaObject


@dataclass(frozen=True)
class MultimodalPrompt:
    """Result of multimodal prompt construction."""

    # Global prefix, carried over from `AdapterSpec`
    global_prefix: str

    # Global suffix, carried over from `AdapterSpec`
    global_suffix: str

    # Instance prefix, carried over from `AdapterSpec`. What goes between the instruction and instances.
    instance_prefix: str

    # Instructions for the task
    instructions: str

    # Train instance blocks for the prompt
    train_instance_blocks: List[MultimediaObject]

    # Evaluation instance
    eval_instance_block: MultimediaObject

    @property
    def multimedia_object(self) -> MultimediaObject:
        """
        Construct the multimodal prompt as a `MultimodalObject` given the train and eval multimodal content
        and adapter spec parameters.
        """
        blocks: List[MultimediaObject] = []
        if self.instructions:
            blocks.append(MultimediaObject([MediaObject(text=self.instructions, content_type="text/plain")]))
        blocks.extend(self.train_instance_blocks)
        blocks.append(self.eval_instance_block)

        # Combine all the content
        # Add the instance prefix between the instruction block and instance blocks if one exists
        result: MultimediaObject = MultimediaObject()
        for i, block in enumerate(blocks):
            result = result.combine(
                block if i == 0 or not self.instance_prefix else block.add_textual_prefix(self.instance_prefix)
            )

        # Add the global prefix if one exists
        if self.global_prefix:
            result = result.add_textual_prefix(self.global_prefix)

        # Add the global prefix if one exists
        if self.global_suffix:
            result = result.add_textual_suffix(self.global_suffix)

        return result

    @property
    def num_train_instances(self) -> int:
        """Returns the number of training instances in the prompt"""
        return len(self.train_instance_blocks)
