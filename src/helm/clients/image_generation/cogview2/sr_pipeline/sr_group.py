# -*- encoding: utf-8 -*-
"""
@File    :   sr_group.py
@Time    :   2022/04/02 01:17:21
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
"""

# here put the import lib
from helm.clients.image_generation.cogview2.sr_pipeline.direct_sr import DirectSuperResolution
from helm.clients.image_generation.cogview2.sr_pipeline.iterative_sr import IterativeSuperResolution

from helm.common.optional_dependencies import handle_module_not_found_error


class SRGroup:
    def __init__(
        self,
        args,
        home_path=None,
    ):
        try:
            from SwissArmyTransformer.resources import auto_create
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        dsr_path = auto_create("cogview2-dsr", path=home_path)
        itersr_path = auto_create("cogview2-itersr", path=home_path)
        dsr = DirectSuperResolution(args, dsr_path)
        itersr = IterativeSuperResolution(args, itersr_path, shared_transformer=dsr.model.transformer)
        self.dsr = dsr
        self.itersr = itersr

    def sr_base(self, img_tokens, txt_tokens):
        assert img_tokens.shape[-1] == 400 and len(img_tokens.shape) == 2
        batch_size = img_tokens.shape[0]
        txt_len = txt_tokens.shape[-1]
        if len(txt_tokens.shape) == 1:
            txt_tokens = txt_tokens.unsqueeze(0).expand(batch_size, txt_len)
        sred_tokens = self.dsr(txt_tokens, img_tokens)
        iter_tokens = self.itersr(txt_tokens, sred_tokens[:, -3600:].clone())
        return iter_tokens[-batch_size:]
