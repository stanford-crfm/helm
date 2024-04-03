from urllib.parse import urljoin


def get_cohere_url(endpoint: str) -> str:
    return urljoin("https://api.cohere.ai", endpoint)
