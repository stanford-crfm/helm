# An example of how to use the request API.

import requests
import json
import getpass
from urllib import parse

auth = {
    'username': 'crfm',
    'password': getpass.getpass(),
}

request = {
    'prompt': 'Life is like a box of',
    'model': 'openai/davinci',
}

params = {
    'auth': json.dumps(auth),
    'request': json.dumps(request),
}

response = requests.get(f'http://crfm-models.stanford.edu/api/makeRequest?' + parse.urlencode(params)).json()
print(json.dumps(response, indent=2))
