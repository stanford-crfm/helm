from dataclasses import dataclass
from typing import List

from helm.common.multimodal_content import MultimodalContent


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
    train_instance_blocks: List[MultimodalContent]

    # Evaluation instance
    eval_instance_block: MultimodalContent

    @property
    def content(self) -> MultimodalContent:
        """Construct the multimodal prompt given the train and eval multimodal content and adapter spec parameters"""
        all_content: List[MultimodalContent] = (
            [MultimodalContent([self.instructions_block])] if self.instructions_block else []
        )
        all_content.extend(self.train_instance_blocks)
        all_content.append(self.eval_instance_block)

        # Combine all the content
        # Add the instance prefix between the instruction block and instance blocks if one exists
        result = MultimodalContent()
        for i, content in enumerate(all_content):
            result = result.combine(
                content if i == 0 or not self.instance_prefix else content.add_textual_prefix(self.instance_prefix)
            )

        # Add the global prefix if one exists
        if self.global_prefix:
            result = result.add_textual_prefix(self.global_prefix)

        return result

    @property
    def num_train_instances(self) -> int:
        """Returns the number of training instances in the prompt"""
        return len(self.train_instance_blocks)
