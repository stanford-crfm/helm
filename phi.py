from helm.clients.vision_language.huggingface_phi_vision_client import HuggingFacePhiVisionClient
from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.common.media_object import MultimediaObject, MediaObject
from helm.common.request import Request


model_name: str = "microsoft/Phi-3.5-vision-instruct"
client = HuggingFacePhiVisionClient(
    tokenizer_name=model_name,
    cache_config=SqliteCacheBackendConfig(path="prod_env/cache").get_cache_config("phi_vision_test"),
)
request = Request(
    model_deployment=model_name,
    model=model_name,
    max_tokens=10,
    multimodal_prompt=MultimediaObject(
        media_objects=[
            MediaObject(
                content_type="image/JPEG",
                location="benchmark_output/scenarios/real_world_qa/0a17edce92135d5b0448cbd605aa1a71.jpg",
            ),
            MediaObject(
                content_type="text/plain",
                text="Write a story about the image.",
            ),
        ]
    ),
)
result = client.make_request(request)
print(result.completions[0].text)
print(f"{result.request_time} seconds")
