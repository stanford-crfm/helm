import getpass

from helm.common.authentication import Authentication
from helm.common.request import RequestResult, TextToImageRequest
from helm.proxy.services.remote_service import RemoteService


# An example of how to use the image generation API.
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("http://localhost:1959")

model: str = "together/StableDiffusion"
request = TextToImageRequest(
    model=model, prompt="A sheep flying over the moon", guidance_scale=7.5, num_completions=3
)
request_result: RequestResult = service.make_request(auth, request)
print(f"Image saved at {request_result.completions[0].file_path} for prompt: {request.prompt}")
