from dataclasses import dataclass, field
from typing import List, Optional, Dict


DEFAULT_ATTRIBUTES = [
    "HATE",
    "HATE/THREATENING",
    "SELF-HARM",
    "SEXUAL",
    "SEXUAL/MINORS",
    "VIOLENCE",
    "VIOLENCE/GRAPHIC"
]

@dataclass(frozen=True)
class ModerationAttributes:
    """
    Moderation attributes and scores calculated by OpenAI. Scores have both a boolean value 
    and a numeric value (0-1).
    """

    # Attribute: HATE
    hate_score: Optional[Dict] = None

    # Attribute: HATE/THREATENING
    hate_threatening_score: Optional[Dict] = None

    # Attribute: SELF-HARM
    self_harm_score: Optional[Dict] = None

    # Attribute: SEXUAL
    sexual_score: Optional[Dict] = None

    # Attribute: SEXUAL/MINORS
    sexual_minors_score: Optional[Dict] = None

    # Attribute: VIOLENCE
    violence_score: Optional[Dict] = None

    # Attribute: VIOLENCE/GRAPHIC
    violence_graphic_score: Optional[Dict] = None


@dataclass(frozen=True)
class OpenAIModerationAPIRequestResult:
    """Result after sending an OpenAIModerationAPIRequest."""

    # Whether the request was successful
    success: bool
    
    # Whether the request was cached
    cached: bool

    # Whether the text was flagged by the API
    flagged: bool
    
    # Moderation attributes for input text
    moderation_attributes: ModerationAttributes 


