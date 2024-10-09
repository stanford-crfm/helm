# mypy: check_untyped_defs = False
import importlib_resources as resources

from helm.common.optional_dependencies import handle_module_not_found_error

import torch

try:
    import sentencepiece as spm
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["yandex"])

"""
From the YaLM GitHub repository (https://github.com/yandex/YaLM-100B),
adapted from https://github.com/yandex/YaLM-100B/blob/main/megatron_lm/megatron/tokenizer/sp_tokenization.py.
"""


YALM_TOKENIZER_PACKAGE: str = "helm.tokenizers.yalm_tokenizer_data"
YALM_TOKENIZER_VOCAB_FILENAME: str = "voc_100b.sp"


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, bytes):
        return text.decode("utf-8")
    elif isinstance(text, str):
        return text
    else:
        raise TypeError(f"Unexpected type {type(text)}")


class YaLMTokenizer:
    NEW_LINE = "[NL]"
    UNK = 0
    BOS = 1
    EOS = 2
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    MASK_TOKEN = "[MASK]"
    MAX_SEQUENCE_LENGTH = 2048

    def __init__(self):
        self.name = "sp"
        vocab_file_path = str(resources.files(YALM_TOKENIZER_PACKAGE).joinpath(YALM_TOKENIZER_VOCAB_FILENAME))
        self._tokenizer = spm.SentencePieceProcessor(model_file=vocab_file_path)
        self._vocab_words = self._get_vocab_words()
        self.encoder = {token: idx for idx, token in enumerate(self._vocab_words)}
        self.decoder = {idx: token for idx, token in enumerate(self._vocab_words)}
        self.padding_side = "left"
        self.truncation_side = "left"

        mask_tokens = self.convert_tokens_to_ids([self.MASK_TOKEN])
        assert len(mask_tokens) == 1
        self.MASK = mask_tokens[0]

    def _encode(self, line, out_type=str):
        return self._tokenizer.encode(line, out_type=out_type)

    def tokenize(self, line, out_type=int):
        line = convert_to_unicode(line)
        line = line.replace("\n", self.NEW_LINE)

        has_bos = False
        has_eos = False

        # Handle special tokens
        if line.startswith(f"{YaLMTokenizer.BOS_TOKEN} "):
            has_bos = True
            line = line[4:]
        elif line.startswith(YaLMTokenizer.BOS_TOKEN):
            has_bos = True
            line = line[3:]
        if line.endswith(f" {YaLMTokenizer.EOS_TOKEN}"):
            has_eos = True
            line = line[:-5]
        elif line.endswith(YaLMTokenizer.EOS_TOKEN):
            has_eos = True
            line = line[:-4]

        token_ids = self._encode(line, out_type=out_type)
        if has_bos:
            if out_type == int:
                token_ids = [1] + token_ids
            else:
                token_ids = [YaLMTokenizer.BOS_TOKEN] + token_ids
        if has_eos:
            if out_type == int:
                token_ids = token_ids + [2]
            else:
                token_ids = token_ids + [YaLMTokenizer.EOS_TOKEN]
        return token_ids

    def convert_tokens_to_ids(self, tokens):
        return self._tokenizer.piece_to_id(tokens)

    def convert_tokens_to_string(self, tokens):
        return self.convert_ids_to_string(self.convert_tokens_to_ids(tokens))

    def convert_ids_to_string(self, ids):
        return [self._tokenizer.decode([i]) for i in ids]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.decoder[ids]
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        return [self.decoder[idx] for idx in ids]

    def get_tokens(self):
        return self._vocab_words

    def _get_vocab_words(self):
        indices = list(range(self._tokenizer.GetPieceSize()))
        return self._tokenizer.id_to_piece(indices)

    @property
    def vocab(self):
        return self.encoder

    @property
    def inv_vocab(self):
        return self.decoder

    @property
    def vocab_size(self):
        return len(self.encoder)

    def detokenize(self, token_ids):
        tokens = [self.decoder[idx] for idx in token_ids]
        text = "".join(tokens).replace("\u2581", " ").lstrip()
        return text

    @property
    def cls(self):
        return self.BOS

    @property
    def eod(self):
        return self.EOS

    @property
    def mask(self):
        return self.MASK

    def __call__(self, text, return_tensors="pt", padding="max_length", truncation=True, add_bos=True):
        assert return_tensors == "pt"
        assert padding == "max_length"

        if isinstance(text, str):
            text = [text]

        ids = []
        for t in text:
            if t.startswith(f"{YaLMTokenizer.BOS_TOKEN} "):
                t_ids = self.tokenize(t[4:])
                t_ids = [1] + t_ids
            elif t.startswith(YaLMTokenizer.BOS_TOKEN):
                t_ids = self.tokenize(t[3:])
                t_ids = [1] + t_ids
            else:
                t_ids = self.tokenize(t)
                if add_bos:
                    t_ids = [1] + t_ids  # append <s>

            if truncation:
                if self.truncation_side == "left":
                    t_ids = t_ids[-self.model_max_length :]
                else:
                    t_ids = t_ids[: self.model_max_length]

            ids.append(t_ids)

        if padding != "max_length":
            max_len = max([len(t_ids) for t_ids in ids])
        else:
            max_len = self.model_max_length

        attention_mask = torch.ones(len(ids), max_len, dtype=torch.long)

        if self.padding_side == "left":
            new_ids = []
            for i, t_ids in enumerate(ids):
                attention_mask[i, : max_len - len(t_ids)] = 0
                new_ids.append([self.BOS] * (max_len - len(t_ids)) + t_ids)
        else:
            new_ids = []
            for i, t_ids in enumerate(ids):
                attention_mask[i, -(max_len - len(t_ids)) :] = 0
                new_ids.append(t_ids + [self.EOS] * (max_len - len(t_ids)))
        ids = new_ids
        ids = torch.tensor(ids)

        if add_bos:
            # make sure starts with <s>
            ids[:, 0] = 1

        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        return self.detokenize(token_ids).replace(self.NEW_LINE, "\n")
