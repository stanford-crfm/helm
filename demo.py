# An example of how to use the request API.

import requests
import json
import getpass
from urllib import parse
from typing import Any, Dict

auth: Dict[str, str] = {
    "username": "crfm",
    "password": getpass.getpass(),
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

response = requests.get(
    f"http://crfm-models.stanford.edu/api/makeRequest?{parse.urlencode(params)}"
).json()
print(json.dumps(response, indent=2))


# Few-shot sentiment classification
request = {
    "prompt": (
        '\nTweet: "I loved the new Batman movie!" Sentiment: positive###'
        '\nTweet: "I hate it when my phone battery dies." Sentiment: negative###'
        '\nTweet: "This is the link to the article" Sentiment: neutral###'
        '\nTweet: "This new music video was okay." Sentiment:'
    ),
    "model": "openai/davinci",
    "temperature": 0.3,
    "maxTokens": 60,
    "topP": 1.0,
    "frequencyPenalty": 0.5,
    "presencePenalty": 0.0,
    "stopSequences": ["###"],
}

params = {
    "auth": json.dumps(auth),
    "request": json.dumps(request),
}

response = requests.get(
    f"http://crfm-models.stanford.edu/api/makeRequest?{parse.urlencode(params)}"
).json()

print(f"Response: {json.dumps(response, indent=2)}")
sentiment: str = response["completions"][0]["text"].strip()
print(f"Output sentiment: {sentiment}")
