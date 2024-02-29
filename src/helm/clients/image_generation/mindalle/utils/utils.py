# ------------------------------------------------------------------------------------
# minDALL-E
# Copyright (c) 2021 Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import random
import urllib
import hashlib
import tarfile
import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm

from helm.common.optional_dependencies import handle_module_not_found_error


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def clip_score(
    prompt: str, images: np.ndarray, model_clip: torch.nn.Module, preprocess_clip, device: str
) -> np.ndarray:
    try:
        import clip
        from PIL import Image
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["heim"])

    images = [preprocess_clip(Image.fromarray((image * 255).astype(np.uint8))) for image in images]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(prompt).to(device=device)
    texts = torch.repeat_interleave(texts, images.shape[0], dim=0)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()
    rank = torch.argsort(scores, descending=True).cpu().numpy()
    return rank


def download(url: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    pathname = filename[: -len(".tar.gz")]

    expected_md5 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    result_path = os.path.join(root, pathname)

    if os.path.isfile(download_target) and (os.path.exists(result_path) and not os.path.isfile(result_path)):
        return result_path

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.md5(open(download_target, "rb").read()).hexdigest() != expected_md5:
        raise RuntimeError(f"Model has been downloaded but the md5 checksum does not not match")

    with tarfile.open(download_target, "r:gz") as f:
        pbar = tqdm(f.getmembers(), total=len(f.getmembers()))
        for member in pbar:
            pbar.set_description(f"extracting: {member.name} (size:{member.size // (1024 * 1024)}MB)")
            f.extract(member=member, path=root)

    return result_path


def realpath_url_or_path(url_or_path: str, root: str = None) -> str:
    if urllib.parse.urlparse(url_or_path).scheme in ("http", "https"):
        return download(url_or_path, root)
    return url_or_path
