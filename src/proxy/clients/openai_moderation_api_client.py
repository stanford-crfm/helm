import os
import requests
from typing import Dict
from dacite import from_dict
import threading
from dataclasses import asdict
from typing import List, Dict, Optional

from googleapiclient.errors import BatchError, HttpError
from googleapiclient.http import BatchHttpRequest
from httplib2 import HttpLib2Error


from common.cache import Cache
from common.general import parse_hocon # TO DO: Remove before final commit
from common.openai_moderation_request import ModerationAttributes, OpenAIModerationAPIRequest, OpenAIModerationAPIRequestResult

CREDENTIALS_FILE = "prod_env/credentials.conf"

class OpenAIModerationAPIClientError(Exception):
    pass

class OpenAIModerationAPIClient:
    """
    OpenAI moderation API predicts whether a string contains content that is not allowed per OpenAI's 
    content moderation policy. It detects seven categories of moderated content and returns a Boolean
    and numeric (0-1) score for each.
    
    Source: https://beta.openai.com/docs/api-reference/moderations/
    """

    @staticmethod
    def extract_moderation_scores(response: Dict) -> ModerationAttributes:
        categories = response["results"][0]["categories"].values()
        category_scores = response["results"][0]["category_scores"].values()
        category_names = response["results"][0]["category_scores"].keys()
        values = [{"is_moderated": x, "score": y} for x, y in list(zip(categories, category_scores))]
        
        all_scores = dict(zip(category_names, values))    
        return from_dict(data_class=ModerationAttributes, data=all_scores)

    def __init__(self, api_key: str, cache_path: str):
        # API key obtained from OpenAI that works with the Moderation model
        self.api_key = api_key

        # Cache requests and responses from Moderation API
        self.cache = Cache(os.path.join(cache_path, "openaimoderationeapi.sqlite"))

    def get_moderation_scores(self, input): #, request: OpenAIModerationAPIRequest) -> OpenAIModerationAPIRequestResult:
        """
        Make call to OpenAI Moderation API. 
        """
        request = {"input": input}

        try:
            def do_it():
                text_to_response: Dict[str, Dict] = dict()

                auth_string = "Bearer "+self.api_key
                headers = {'content-type': 'application/json', "Authorization": auth_string}
                response = requests.post(url = "https://api.openai.com/v1/moderations", json=request, headers=headers)
                
                text_to_response[input] = response

                return text_to_response

        except Exception as e:
            raise OpenAIModerationAPIClientError(f"Error was thrown when making a request to OpenAI Moderation API: ", e)
        
        response, cached = self.cache.get(request, do_it)
        response_json = response[input].json()

        moderation_attributes = OpenAIModerationAPIClient.extract_moderation_scores(response_json)
        
        return OpenAIModerationAPIRequestResult(
            cached=cached,
            flagged=response_json["results"][0]["flagged"],
            text_to_moderation_attributes=moderation_attributes
        )
    
    def test_moderation_class(self):
        """
        Temporary fn for debugging, will delete this + main when making final commit
        """
        result = self.get_moderation_scores("This is a test")
        print(result)
    

def main():
    """
    Temporary for debug + test, will remove in final commit
    """
    credentials_path = CREDENTIALS_FILE
    with open(credentials_path) as f:
        credentials = parse_hocon(f.read())
    
    api_key = credentials["openaiApiKey"]
    test_client = OpenAIModerationAPIClient(api_key=api_key, cache_path=".")
    test_client.test_moderation_class()

if __name__ == "__main__":
    main()

