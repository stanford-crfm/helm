from typing import List
from helm.common.cache_backend_config import BlackHoleCacheBackendConfig

from helm.common.request import Request, GeneratedOutput, Token
from helm.tokenizers.auto_tokenizer import AutoTokenizer
from helm.proxy.token_counters.auto_token_counter import AutoTokenCounter


class TestAutoTokenCounter:
    def test_count_tokens_openai(self):
        token_counter = AutoTokenCounter(
            AutoTokenizer(credentials={}, cache_backend_config=BlackHoleCacheBackendConfig())
        )
        # The following prompt has 51 tokens according to the GPT-2 tokenizer
        request = Request(
            model="openai/gpt2",
            model_deployment="huggingface/gpt2",
            prompt="The Center for Research on Foundation Models (CRFM) is "
            "an interdisciplinary initiative born out of the Stanford "
            "Institute for Human-Centered Artificial Intelligence (HAI) "
            "that aims to make fundamental advances in the study, development, "
            "and deployment of foundation models.",
        )
        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text=" The CRFM is dedicated to advancing our knowledge of the foundations of artificial intelligence "
                "(AI) and related fields. It focuses on foundational questions in AI, which are",
                logprob=-49.00783279519999,
                tokens=[
                    Token(text=" The", logprob=-1.8096403),
                    Token(text=" CR", logprob=-1.2861944),
                    Token(text="FM", logprob=-0.0032369632),
                    Token(text=" is", logprob=-1.4355252),
                    Token(text=" dedicated", logprob=-3.814422),
                    Token(text=" to", logprob=-0.009623392),
                    Token(text=" advancing", logprob=-2.6732886),
                    Token(text=" our", logprob=-3.123714),
                    Token(text=" knowledge", logprob=-3.030337),
                    Token(text=" of", logprob=-0.46280858),
                    Token(text=" the", logprob=-1.4058315),
                    Token(text=" foundations", logprob=-2.0638132),
                    Token(text=" of", logprob=-0.2607486),
                    Token(text=" artificial", logprob=-1.1653417),
                    Token(text=" intelligence", logprob=-0.03756146),
                    Token(text=" (", logprob=-2.019812),
                    Token(text="AI", logprob=-0.03869382),
                    Token(text=")", logprob=-0.49895737),
                    Token(text=" and", logprob=-0.81909865),
                    Token(text=" related", logprob=-2.611718),
                    Token(text=" fields", logprob=-0.7640527),
                    Token(text=".", logprob=-1.8066244),
                    Token(text=" It", logprob=-2.2856107),
                    Token(text=" focuses", logprob=-3.3735154),
                    Token(text=" on", logprob=-0.13244776),
                    Token(text=" foundational", logprob=-1.2640914),
                    Token(text=" questions", logprob=-2.010647),
                    Token(text=" in", logprob=-1.980726),
                    Token(text=" AI", logprob=-0.5709368),
                    Token(text=",", logprob=-1.036094),
                    Token(text=" which", logprob=-3.826836),
                    Token(text=" are", logprob=-1.3858839),
                ],
            )
        ]

        # Verified against https://beta.openai.com/tokenizer. Prompt + completions = 51 + 32.
        assert token_counter.count_tokens(request, completions) == 51 + 32

    def test_count_tokens_anthropic(self):
        token_counter = AutoTokenCounter(
            AutoTokenizer(credentials={}, cache_backend_config=BlackHoleCacheBackendConfig())
        )
        request = Request(
            model="anthropic/claude-instant-v1",
            model_deployment="anthropic/claude-instant-v1",
            prompt="\n\nHuman:The Center for Research on Foundation Models (CRFM) is "
            "an interdisciplinary initiative born out of the Stanford "
            "Institute for Human-Centered Artificial Intelligence (HAI) "
            "that aims to make fundamental advances in the study, development, "
            "and deployment of foundation models.\n\nAssistant:",
        )
        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="Thank you for the background information. The Center for Research "
                "on Foundation Models sounds like an interesting initiative focused on "
                "advancing research and responsible development of large AI models. I "
                "don't have any personal thoughts on it, but I'm happy to discuss or "
                "provide more information if helpful. As an AI assistant, I don't have "
                "subjective opinions.",
                logprob=0,
                tokens=[
                    Token(text="Thank", logprob=0),
                    Token(text=" you", logprob=0),
                    Token(text=" for", logprob=0),
                    Token(text=" the", logprob=0),
                    Token(text=" background", logprob=0),
                    Token(text=" information", logprob=0),
                    Token(text=".", logprob=0),
                    Token(text=" The", logprob=0),
                    Token(text=" Center", logprob=0),
                    Token(text=" for", logprob=0),
                    Token(text=" Research", logprob=0),
                    Token(text=" on", logprob=0),
                    Token(text=" Foundation", logprob=0),
                    Token(text=" Models", logprob=0),
                    Token(text=" sounds", logprob=0),
                    Token(text=" like", logprob=0),
                    Token(text=" an", logprob=0),
                    Token(text=" interesting", logprob=0),
                    Token(text=" initiative", logprob=0),
                    Token(text=" focused", logprob=0),
                    Token(text=" on", logprob=0),
                    Token(text=" advancing", logprob=0),
                    Token(text=" research", logprob=0),
                    Token(text=" and", logprob=0),
                    Token(text=" responsible", logprob=0),
                    Token(text=" development", logprob=0),
                    Token(text=" of", logprob=0),
                    Token(text=" large", logprob=0),
                    Token(text=" AI", logprob=0),
                    Token(text=" models", logprob=0),
                    Token(text=".", logprob=0),
                    Token(text=" I", logprob=0),
                    Token(text=" don", logprob=0),
                    Token(text="'t", logprob=0),
                    Token(text=" have", logprob=0),
                    Token(text=" any", logprob=0),
                    Token(text=" personal", logprob=0),
                    Token(text=" thoughts", logprob=0),
                    Token(text=" on", logprob=0),
                    Token(text=" it", logprob=0),
                    Token(text=",", logprob=0),
                    Token(text=" but", logprob=0),
                    Token(text=" I", logprob=0),
                    Token(text="'m", logprob=0),
                    Token(text=" happy", logprob=0),
                    Token(text=" to", logprob=0),
                    Token(text=" discuss", logprob=0),
                    Token(text=" or", logprob=0),
                    Token(text=" provide", logprob=0),
                    Token(text=" more", logprob=0),
                    Token(text=" information", logprob=0),
                    Token(text=" if", logprob=0),
                    Token(text=" helpful", logprob=0),
                    Token(text=".", logprob=0),
                    Token(text=" As", logprob=0),
                    Token(text=" an", logprob=0),
                    Token(text=" AI", logprob=0),
                    Token(text=" assistant", logprob=0),
                    Token(text=",", logprob=0),
                    Token(text=" I", logprob=0),
                    Token(text=" don", logprob=0),
                    Token(text="'t", logprob=0),
                    Token(text=" have", logprob=0),
                    Token(text=" subjective", logprob=0),
                    Token(text=" opinions", logprob=0),
                    Token(text=".", logprob=0),
                ],
                finish_reason=None,
                multimodal_content=None,
            )
        ]

        assert token_counter.count_tokens(request, completions) == 126
