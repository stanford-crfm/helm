""" DalleBart tokenizer """

from transformers import BartTokenizerFast

from helm.clients.image_generation.dalle_mini.model.utils import PretrainedFromWandbMixin


class DalleBartTokenizer(PretrainedFromWandbMixin, BartTokenizerFast):
    pass
