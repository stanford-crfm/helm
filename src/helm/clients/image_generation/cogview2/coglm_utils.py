import torch

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from SwissArmyTransformer.model import CachedAutoregressiveModel
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["heim"])


def get_masks_and_position_ids_coglm(seq, context_length):
    tokens = seq.unsqueeze(0)
    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)
    position_ids = torch.zeros(len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[:context_length])
    torch.arange(512, 512 + len(seq) - context_length, out=position_ids[context_length:])
    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


def get_recipe(name):
    r = {
        "attn_plus": 1.4,
        "temp_all_gen": 1.15,
        "topk_gen": 16,
        "temp_cluster_gen": 1.0,
        "temp_all_dsr": 1.5,
        "topk_dsr": 100,
        "temp_cluster_dsr": 0.89,
        "temp_all_itersr": 1.3,
        "topk_itersr": 16,
        "query_template": "{}<start_of_image>",
    }
    if name == "none":
        pass
    elif name == "mainbody":
        r["query_template"] = "{} 高清摄影 隔绝<start_of_image>"
    elif name == "photo":
        r["query_template"] = "{} 高清摄影<start_of_image>"
    elif name == "flat":
        r["query_template"] = "{} 平面风格<start_of_image>"
        # r['attn_plus'] = 1.8
        # r['temp_cluster_gen'] = 0.75
        r["temp_all_gen"] = 1.1
        r["topk_dsr"] = 5
        r["temp_cluster_dsr"] = 0.4
        r["temp_all_itersr"] = 1
        r["topk_itersr"] = 5
    elif name == "comics":
        r["query_template"] = "{} 漫画 隔绝<start_of_image>"
        r["topk_dsr"] = 5
        r["temp_cluster_dsr"] = 0.4
        r["temp_all_gen"] = 1.1
        r["temp_all_itersr"] = 1
        r["topk_itersr"] = 5
    elif name == "oil":
        r["query_template"] = "{} 油画风格<start_of_image>"
        pass
    elif name == "sketch":
        r["query_template"] = "{} 素描风格<start_of_image>"
        r["temp_all_gen"] = 1.1
    elif name == "isometric":
        r["query_template"] = "{} 等距矢量图<start_of_image>"
        r["temp_all_gen"] = 1.1
    elif name == "chinese":
        r["query_template"] = "{} 水墨国画<start_of_image>"
        r["temp_all_gen"] = 1.12
    elif name == "watercolor":
        r["query_template"] = "{} 水彩画风格<start_of_image>"
    return r


class InferenceModel(CachedAutoregressiveModel):
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float()
        )
        return logits_parallel
