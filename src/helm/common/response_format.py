from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ResponseFormat:
    """EXPERIMENTAL: Model response format.

    Currently only supports JSON schema.

    Currently only supported by OpenAI and Together.

    See:
    - https://platform.openai.com/docs/guides/structured-outputs
    - https://docs.together.ai/docs/json-mode"""

    json_schema: Optional[Dict[str, Any]] = None
    """EXPERIMENTAL: The JSON schema that the model output should conform to."""
