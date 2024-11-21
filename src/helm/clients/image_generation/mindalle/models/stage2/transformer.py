# ------------------------------------------------------------------------------------
# minDALL-E
# Copyright (c) 2021 Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from minGPT (https://github.com/karpathy/minGPT)
# Copyright (c) 2020 Andrej Karpathy. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch.cuda.amp import autocast
from helm.clients.image_generation.mindalle.models.stage2.layers import Block

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["heim"])


class Transformer1d(nn.Module):
    def __init__(self, vocab_size_txt: int, vocab_size_img: int, hparams: OmegaConf) -> None:
        super().__init__()
        assert hparams.n_layers == hparams.n_dense_layers

        # input embedding for image and text
        self.tok_emb_img = nn.Embedding(vocab_size_img, hparams.embed_dim)
        self.tok_emb_txt = nn.Embedding(vocab_size_txt, hparams.embed_dim)

        self.pos_emb_img = nn.Embedding(hparams.ctx_len_img, hparams.embed_dim)
        self.pos_emb_txt = nn.Embedding(hparams.ctx_len_txt, hparams.embed_dim)

        self.drop = nn.Dropout(hparams.embd_pdrop)

        # transformer blocks
        self.blocks = [
            Block(
                ctx_len=hparams.ctx_len_img + hparams.ctx_len_txt,
                embed_dim=hparams.embed_dim,
                n_heads=hparams.n_heads,
                mlp_bias=hparams.mlp_bias,
                attn_bias=hparams.attn_bias,
                resid_pdrop=hparams.resid_pdrop,
                attn_pdrop=hparams.attn_pdrop,
                gelu_use_approx=hparams.gelu_use_approx,
            )
            for i in range(1, hparams.n_layers + 1)
        ]
        self.blocks = nn.Sequential(*self.blocks)

        # heads for image and text
        self.ln_f = nn.LayerNorm(hparams.embed_dim)
        self.head_img = nn.Linear(hparams.embed_dim, vocab_size_img, bias=False)
        self.head_txt = nn.Linear(hparams.embed_dim, vocab_size_txt, bias=False)

        self.ctx_len_img = hparams.ctx_len_img
        self.ctx_len_txt = hparams.ctx_len_txt
        self.n_layers = hparams.n_layers

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        images: torch.LongTensor,
        texts: torch.LongTensor,
        pos_images: torch.LongTensor,
        pos_texts: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        B, T = images.shape
        _, N = texts.shape

        assert T <= self.ctx_len_img, "Already reached the maximum context length (image)."
        assert N == self.ctx_len_txt, "Already reached the maximum context length (text)."

        texts = self.tok_emb_txt(texts)
        images = self.tok_emb_img(images)

        texts = texts + self.pos_emb_txt(pos_texts)
        images = images + self.pos_emb_img(pos_images)

        x = torch.cat([texts, images], axis=1).contiguous()
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        texts = x[:, : N - 1].contiguous()
        images = x[:, N - 1 : -1].contiguous()

        logits_txt = self.head_txt(texts)
        logits_img = self.head_img(images)
        return logits_img, logits_txt

    @torch.no_grad()
    def sampling(
        self,
        images: torch.LongTensor,
        texts: torch.LongTensor,
        pos_images: torch.LongTensor,
        pos_texts: torch.LongTensor,
        use_fp16: bool = True,
        past: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        _, N = texts.shape
        assert N == self.ctx_len_txt, "Already reached the maximum context length (text)."

        with autocast(enabled=use_fp16):
            if images is None:
                assert past is None

                texts = self.tok_emb_txt(texts)
                x = texts + self.pos_emb_txt(pos_texts)
                x = self.drop(x)

                presents = []
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.ln_f(x)
                x = x[:, N - 1].contiguous()
                logits = self.head_img(x)
            else:
                if past is None:
                    texts = self.tok_emb_txt(texts)
                    images = self.tok_emb_img(images)
                    texts = texts + self.pos_emb_txt(pos_texts)
                    images = images + self.pos_emb_img(pos_images)
                    x = torch.cat([texts, images], axis=1).contiguous()
                else:
                    images = self.tok_emb_img(images)
                    x = images + self.pos_emb_img(pos_images)
                x = self.drop(x)

                if past is not None:
                    past = torch.cat(past, dim=-2)
                presents = []
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=None if past is None else past[i])
                    presents.append(present)
                x = self.ln_f(x)
                x = x[:, -1].contiguous()
                logits = self.head_img(x)
            return logits, presents

    def from_ckpt(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(ckpt, strict=True)
        print(f"{path} succesfully restored..")


class iGPT(nn.Module):
    def __init__(self, vocab_size_img: int, use_cls_cond: bool, hparams: OmegaConf) -> None:
        super().__init__()
        self.use_cls_cond = use_cls_cond

        # sos token embedding
        if self.use_cls_cond:
            self.sos = nn.Embedding(hparams.n_classes, hparams.embed_dim)
        else:
            self.sos = nn.Parameter(torch.randn(1, 1, hparams.embed_dim))

        # input embedding
        self.tok_emb_img = nn.Embedding(vocab_size_img, hparams.embed_dim)
        self.pos_emb_img = nn.Embedding(hparams.ctx_len_img, hparams.embed_dim)

        self.drop = nn.Dropout(hparams.embd_pdrop)

        # transformer blocks
        self.blocks = [
            Block(
                ctx_len=hparams.ctx_len_img + 1,
                embed_dim=hparams.embed_dim,
                n_heads=hparams.n_heads,
                mlp_bias=hparams.mlp_bias,
                attn_bias=hparams.attn_bias,
                resid_pdrop=hparams.resid_pdrop,
                attn_pdrop=hparams.attn_pdrop,
                gelu_use_approx=hparams.gelu_use_approx,
            )
            for i in range(1, hparams.n_layers + 1)
        ]
        self.blocks = nn.Sequential(*self.blocks)

        # head
        self.ln_f = nn.LayerNorm(hparams.embed_dim)
        self.head = nn.Linear(hparams.embed_dim, vocab_size_img, bias=False)

        self.ctx_len_img = hparams.ctx_len_img
        self.n_layers = hparams.n_layers

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @torch.no_grad()
    def sampling(
        self,
        sos: torch.FloatTensor,
        codes: torch.LongTensor,
        pos_codes: torch.LongTensor,
        n_samples: int = 16,
        use_fp16: bool = True,
        past: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        with autocast(enabled=use_fp16):
            if codes is None:
                assert past is None
                xs = self.drop(sos)
                presents = []
                for i, block in enumerate(self.blocks):
                    xs, present = block.sample(xs, layer_past=None)
                    presents.append(present)
                xs = self.ln_f(xs)
                logits = self.head(xs)[:, -1]
            else:
                if past is None:
                    xs = self.tok_emb_img(codes) + self.pos_emb_img(pos_codes)
                    xs = torch.cat([sos, xs], dim=1)
                else:
                    xs = self.tok_emb_img(codes) + self.pos_emb_img(pos_codes)
                xs = self.drop(xs)

                past = torch.cat(past, dim=-2) if past is not None else past
                presents = []
                for i, block in enumerate(self.blocks):
                    xs, present = block.sample(xs, layer_past=None if past is None else past[i])
                    presents.append(present)

                xs = self.ln_f(xs)
                logits = self.head(xs)[:, -1]
            return logits, presents

    def forward(self, codes: torch.LongTensor, labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        B, T = codes.shape
        xps = torch.arange(T, device=codes.device).repeat((B, 1))
        sos = self.sos.repeat((B, 1, 1)) if labels is None else self.sos(labels).unsqueeze(1)

        h = self.tok_emb_img(codes) + self.pos_emb_img(xps)
        h = torch.cat([sos, h[:, :-1]], dim=1).contiguous()

        h = self.drop(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def from_ckpt(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(ckpt, strict=strict)
        print(f"{path} successfully restored..")
