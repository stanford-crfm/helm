from typing import List

from helm.common.request import Request, Sequence, Token
from .ai21_token_counter import AI21TokenCounter


class TestAI21TokenCounter:
    def setup_method(self, method):
        self.token_counter = AI21TokenCounter()

    def test_count_tokens(self):
        request = Request(
            model="openai/text-davinci-002",
            model_deployment="openai/text-davinci-002",
            prompt="The Center for Research on Foundation Models (CRFM) is "
            "an interdisciplinary initiative born out of the Stanford "
            "Institute for Human-Centered Artificial Intelligence (HAI) "
            "that aims to make fundamental advances in the study, development, "
            "and deployment of foundation models.",
        )
        completions: List[Sequence] = [
            Sequence(
                text="\n\nFoundation models are an artificial intelligence paradigm that emphasizes: (1) reasoning "
                "about symbolic, structured knowledge, (2) learning to perform tasks from observation, ",
                logprob=-66.24831021729915,
                tokens=[
                    Token(text="\n", logprob=-1.034758448600769),
                    Token(text="\n", logprob=-2.325321674346924),
                    Token(
                        text=" Foundation",
                        logprob=-1.2575088739395142,
                    ),
                    Token(
                        text=" models are",
                        logprob=-0.9496442079544067,
                    ),
                    Token(text=" an", logprob=-5.785625457763672),
                    Token(
                        text=" artificial intelligence",
                        logprob=-2.073937177658081,
                    ),
                    Token(text=" paradigm", logprob=-2.855092763900757),
                    Token(text=" that", logprob=-1.380897879600525),
                    Token(
                        text=" emphasizes",
                        logprob=-4.230795860290527,
                    ),
                    Token(text=":", logprob=-5.380178451538086),
                    Token(text=" ", logprob=-1.1408348083496094),
                    Token(text="(", logprob=-0.41460439562797546),
                    Token(text="1", logprob=-0.5666008591651917),
                    Token(text=")", logprob=-0.001801535952836275),
                    Token(text=" reasoning", logprob=-3.4144058227539062),
                    Token(text=" about", logprob=-1.3604949712753296),
                    Token(text=" symbolic", logprob=-7.108627796173096),
                    Token(text=",", logprob=-2.8421378135681152),
                    Token(
                        text=" structured",
                        logprob=-2.6082611083984375,
                    ),
                    Token(text=" knowledge", logprob=-0.91008061170578),
                    Token(text=",", logprob=-1.0750247240066528),
                    Token(text=" ", logprob=-0.5834965705871582),
                    Token(text="(", logprob=-0.0004963834653608501),
                    Token(text="2", logprob=-0.0009141556802205741),
                    Token(text=")", logprob=-5.686121585313231e-05),
                    Token(text=" learning", logprob=-2.123058319091797),
                    Token(text=" to perform", logprob=-5.197870254516602),
                    Token(text=" tasks", logprob=-1.5782833099365234),
                    Token(text=" from", logprob=-1.1503676176071167),
                    Token(text=" observation", logprob=-4.8489789962768555),
                    Token(text=",", logprob=-0.7239797711372375),
                    Token(text=" ", logprob=-1.3241727352142334),
                ],
            )
        ]

        # Verified against https://studio.ai21.com/playground.
        assert self.token_counter.count_tokens(request, completions) == 32
