from typing import Dict

from client import Client
from openai_client import OpenAIClient
from ai21_client import AI21Client
from schemas import Request, RequestResult
from simple_client import SimpleClient

class AutoClient(Client):
    """Automatically dispatch to the proper client."""
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.clients: Dict[str, Client] = {}

    def get_client(self, organization: str) -> Client:
        """Return a client based on `organization`, creating it if necessary."""
        client = self.clients.get(organization)
        if client is None:
            if organization == 'openai':
                client = OpenAIClient(api_key=self.credentials['openaiApiKey'])
            elif organization == 'ai21':
                client = AI21Client(api_key=self.credentials['ai21ApiKey'])
            elif organization == 'huggingface':
                client = HuggingFaceClient()
            elif organization == 'simple':
                client = SimpleClient()
            else:
                raise Exception(f'Unknown organization: {organization}')
            self.clients[organization] = client
        return client

    def make_request(self, request: Request) -> RequestResult:
        # Dispatch based on the organization (e.g., openai/davinci)
        organization = request.model_organization()
        client = self.get_client(organization)
        return client.make_request(request)
