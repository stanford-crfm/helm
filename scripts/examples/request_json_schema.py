"""Example of using a JSON Schema with a request"""

from helm.clients.auto_client import AutoClient
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from helm.common.request import Request, ResponseFormat
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package

register_builtin_configs_from_helm_package()

client = AutoClient({}, ".", BlackHoleCacheBackendConfig())

json_schema = {
    "properties": {
        "title": {"description": "A title for the voice note", "title": "Title", "type": "string"},
        "summary": {
            "description": "A short one sentence summary of the voice note.",
            "title": "Summary",
            "type": "string",
        },
        "actionItems": {
            "description": "A list of action items from the voice note",
            "items": {"type": "string"},
            "title": "Actionitems",
            "type": "array",
        },
    },
    "required": ["title", "summary", "actionItems"],
    "title": "VoiceNote",
    "type": "object",
}

# Normally, you would want to explain the expected output format in the prompt instructions,
# including a list of field names, types, and descriptions.
# In this demo, the explanation is omitted in the demo prompt in order to demonstrate that the fields are
# induced solely by the JSON schema.
prompt = """Extract information from the following voice message transcript. Answer only with a JSON object.

Good morning! It's 7:00 AM, and I'm just waking up. Today is going to be a busy day, so let's get started. First, I need to make a quick breakfast. I think I'll have some scrambled eggs and toast with a cup of coffee. While I'm cooking, I'll also check my emails to see if there's anything urgent."""  # noqa: E501

request = Request(
    model_deployment="openai/gpt-4o-2024-11-20",
    model="openai/gpt-4o-2024-11-20",
    embedding=False,
    prompt=prompt,
    temperature=0.0,
    num_completions=1,
    top_k_per_token=1,
    max_tokens=1000,
    response_format=ResponseFormat(json_schema=json_schema),
)

response = client.make_request(request)
print(response.completions[0].text)
