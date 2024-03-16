from urllib.parse import urljoin


# From "https://docs.cohere.ai/versioning-reference",
# "this version [2021-11-08] introduces multiple generations, meaning that the generations endpoint will
# now accept a num_generations argument in the JSON and will always return an array of generations"
# Note that the API version is decoupled from the model version.
DEFAULT_COHERE_API_VERSION: str = "2021-11-08"


def get_cohere_url(endpoint: str) -> str:
    return urljoin("https://api.cohere.ai", endpoint)
