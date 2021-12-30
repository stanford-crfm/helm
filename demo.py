# An example of how to use the request API.

import requests
import json
from urllib import parse
from typing import Any, Dict

auth: Dict[str, str] = {
    "api_key": "crfm",
}

# Generation
request: Dict[str, Any] = {
    "prompt": "Life is like a box of",
    "model": "openai/davinci",
}

params = {
    "auth": json.dumps(auth),
    "request": json.dumps(request),
}

response = requests.get(f"http://crfm-models.stanford.edu/api/request?{parse.urlencode(params)}").json()
print(json.dumps(response, indent=2))
