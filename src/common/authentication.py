from dataclasses import dataclass


@dataclass(frozen=True)
class Authentication:
    """Needed to authenticate with the proxy server to make requests, etc.."""

    api_key: str
