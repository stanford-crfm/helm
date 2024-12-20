# ------------------------------------------------------------------------------------
# Modified from VQGAN (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from helm.clients.image_generation.mindalle.models.stage1.layers import Encoder, Decoder
from helm.common.optional_dependencies import handle_module_not_found_error


class VectorQuantizer(nn.Module):
    """
    Simplified VectorQuantizer in the original VQGAN repository
    by removing unncessary modules for sampling
    """

    def __init__(self, dim: int, n_embed: int, beta: float) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embed, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        try:
            from einops import rearrange
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        z = rearrange(z, "b c h w -> b h w c").contiguous()  # [B,C,H,W] -> [B,H,W,C]
        z_flattened = z.view(-1, self.dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Optional[List[int]] = None) -> torch.FloatTensor:
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class VQGAN(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, hparams) -> None:
        super().__init__()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.quantize = VectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(hparams.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hparams.z_channels, 1)
        self.latent_dim = hparams.attn_resolutions[0]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.encode(x)
        dec = self.decode(quant)
        return dec

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        try:
            from einops import rearrange
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        h = self.encoder(x)
        h = self.quant_conv(h)
        quant = self.quantize(h)[0]
        quant = rearrange(quant, "b h w c -> b c h w").contiguous()
        return quant

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantize.get_codebook_entry(code)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def get_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        codes = self.quantize(h)[1].view(x.shape[0], self.latent_dim**2)
        return codes

    def from_ckpt(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(ckpt, strict=strict)
        print(f"{path} successfully restored..")
