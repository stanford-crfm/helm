from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ImageGenerationParameters:
    """
    Parameters for image generation.
    """

    output_image_width: Optional[int] = None
    """Width of the generated image. The model will generate images with the model's
    default dimensions when unspecified."""

    output_image_height: Optional[int] = None
    """Height of the generated image. The model will generate images with the model's
    default dimensions when unspecified."""

    guidance_scale: Optional[float] = None
    """A non-negative number determining how much importance is given to the prompt
    when generating images. Higher values will generate images that follow more
    closely to the prompt. Currently only for diffusion models."""

    diffusion_denoising_steps: Optional[int] = None
    """The number of denoising steps for diffusion models."""
