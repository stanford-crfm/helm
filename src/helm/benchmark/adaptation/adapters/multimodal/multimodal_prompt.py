from dataclasses import dataclass
from typing import List

from helm.common.media_object import MediaObject, add_textual_prefix


@dataclass(frozen=True)
class MultimodalPrompt:
    """Result of multimodal prompt construction."""

    # Global prefix, carried over from `AdapterSpec`
    global_prefix: str

    # Instance prefix, carried over from `AdapterSpec`. What goes between the instruction and instances.
    instance_prefix: str

    # Instructions for the task
    instructions_block: str

    # Train instance blocks for the prompt
    train_instance_blocks: List[List[MediaObject]]

    # Evaluation instance
    eval_instance_block: List[MediaObject]

    @property
    def content(self) -> List[MediaObject]:
        """Construct the multimodal prompt given the train and eval multimodal content and adapter spec parameters"""
        blocks: List[List[MediaObject]] = (
            [[MediaObject(text=self.instructions_block, content_type="text/plain")]] if self.instructions_block else []
        )
        blocks.extend(self.train_instance_blocks)
        blocks.append(self.eval_instance_block)

        # Combine all the content
        # Add the instance prefix between the instruction block and instance blocks if one exists
        result: List[MediaObject] = []
        for i, block in enumerate(blocks):
            result.extend(
                block if i == 0 or not self.instance_prefix else add_textual_prefix(block, self.instance_prefix)
            )

        # Add the global prefix if one exists
        if self.global_prefix:
            result = add_textual_prefix(result, self.global_prefix)

        return result

    @property
    def num_train_instances(self) -> int:
        """Returns the number of training instances in the prompt"""
        return len(self.train_instance_blocks)
