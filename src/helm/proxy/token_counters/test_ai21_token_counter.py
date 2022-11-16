from typing import List

from helm.common.request import Request, Sequence, Token
from .ai21_token_counter import AI21TokenCounter


class TestAI21TokenCounter:
    def setup_method(self, method):
        self.token_counter = AI21TokenCounter()

    def test_count_tokens(self):
        request = Request(
            prompt="The Center for Research on Foundation Models (CRFM) is "
            "an interdisciplinary initiative born out of the Stanford "
            "Institute for Human-Centered Artificial Intelligence (HAI) "
            "that aims to make fundamental advances in the study, development, "
            "and deployment of foundation models."
        )
        completions: List[Sequence] = [
            Sequence(
                text="\n\nFoundation models are an artificial intelligence paradigm that emphasizes: (1) reasoning "
                "about symbolic, structured knowledge, (2) learning to perform tasks from observation, ",
                logprob=-66.24831021729915,
                tokens=[
                    Token(text="\n", logprob=-1.034758448600769, top_logprobs={"\n": -1.034758448600769}),
                    Token(text="\n", logprob=-2.325321674346924, top_logprobs={" Foundation": -1.2628217935562134}),
                    Token(
                        text=" Foundation",
                        logprob=-1.2575088739395142,
                        top_logprobs={" Foundation": -1.2575088739395142},
                    ),
                    Token(
                        text=" models are",
                        logprob=-0.9496442079544067,
                        top_logprobs={" models are": -0.9496442079544067},
                    ),
                    Token(text=" an", logprob=-5.785625457763672, top_logprobs={" a class of": -2.762187957763672}),
                    Token(
                        text=" artificial intelligence",
                        logprob=-2.073937177658081,
                        top_logprobs={" increasingly popular": -1.714562177658081},
                    ),
                    Token(text=" paradigm", logprob=-2.855092763900757, top_logprobs={" ": -1.2613427639007568}),
                    Token(text=" that", logprob=-1.380897879600525, top_logprobs={" that": -1.380897879600525}),
                    Token(
                        text=" emphasizes",
                        logprob=-4.230795860290527,
                        top_logprobs={" attempts to": -3.5276708602905273},
                    ),
                    Token(text=":", logprob=-5.380178451538086, top_logprobs={" reasoning": -2.192678689956665}),
                    Token(text=" ", logprob=-1.1408348083496094, top_logprobs={"\n": -0.6095848083496094}),
                    Token(text="(", logprob=-0.41460439562797546, top_logprobs={"(": -0.41460439562797546}),
                    Token(text="1", logprob=-0.5666008591651917, top_logprobs={"1": -0.5666008591651917}),
                    Token(text=")", logprob=-0.001801535952836275, top_logprobs={")": -0.001801535952836275}),
                    Token(text=" reasoning", logprob=-3.4144058227539062, top_logprobs={" the": -2.3987808227539062}),
                    Token(text=" about", logprob=-1.3604949712753296, top_logprobs={" about": -1.3604949712753296}),
                    Token(text=" symbolic", logprob=-7.108627796173096, top_logprobs={" and": -2.5617527961730957}),
                    Token(text=",", logprob=-2.8421378135681152, top_logprobs={" knowledge": -1.6233878135681152}),
                    Token(
                        text=" structured",
                        logprob=-2.6082611083984375,
                        top_logprobs={" structured": -2.6082611083984375},
                    ),
                    Token(text=" knowledge", logprob=-0.91008061170578, top_logprobs={" knowledge": -0.91008061170578}),
                    Token(text=",", logprob=-1.0750247240066528, top_logprobs={",": -1.0750247240066528}),
                    Token(text=" ", logprob=-0.5834965705871582, top_logprobs={" ": -0.5834965705871582}),
                    Token(text="(", logprob=-0.0004963834653608501, top_logprobs={"(": -0.0004963834653608501}),
                    Token(text="2", logprob=-0.0009141556802205741, top_logprobs={"2": -0.0009141556802205741}),
                    Token(text=")", logprob=-5.686121585313231e-05, top_logprobs={")": -5.686121585313231e-05}),
                    Token(text=" learning", logprob=-2.123058319091797, top_logprobs={" learning": -2.123058319091797}),
                    Token(
                        text=" to perform", logprob=-5.197870254516602, top_logprobs={" through": -1.7916204929351807}
                    ),
                    Token(text=" tasks", logprob=-1.5782833099365234, top_logprobs={" complex": -1.5470333099365234}),
                    Token(text=" from", logprob=-1.1503676176071167, top_logprobs={" from": -1.1503676176071167}),
                    Token(
                        text=" observation", logprob=-4.8489789962768555, top_logprobs={" human": -2.5989787578582764}
                    ),
                    Token(text=",", logprob=-0.7239797711372375, top_logprobs={",": -0.7239797711372375}),
                    Token(text=" ", logprob=-1.3241727352142334, top_logprobs={" and": -0.5897976756095886}),
                ],
            )
        ]

        # Verified against https://studio.ai21.com/playground.
        assert self.token_counter.count_tokens(request, completions) == 32
