from proxy.huggingface_client import HuggingFaceClient
from common.request import Request


if __name__ == "__main__":
    client = HuggingFaceClient(cache_path="huggingface_cache")
    print(
        client.make_request(
            Request(
                model="huggingface/gptj_6b",
                prompt="I am a computer scientist.",
                num_completions=2,
                max_tokens=20,
                top_k_per_token=5,
            )
        )
    )
    print(
        client.make_request(
            Request(
                model="huggingface/gpt2",
                prompt="I am a computer scientist.",
                num_completions=2,
                max_tokens=20,
                top_k_per_token=5,
            )
        )
    )
    print(
        client.make_request(
            Request(
                model="huggingface/gpt2",
                prompt="I am a computer scientist.",
                num_completions=2,
                max_tokens=20,
                top_k_per_token=5,
                echo_prompt=True,
            )
        )
    )
