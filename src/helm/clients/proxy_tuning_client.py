# File: helm/clients/proxy_tuning_client.py
from helm.clients.client import Client
from helm.common.request import Request, RequestResult, GeneratedOutput

from typing import Optional, Dict, Any, List
import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
)
import tqdm
from transformers import BitsAndBytesConfig
import math
from typing import Literal
from datetime import datetime

MODEL_PATHS = {
    "llama-70b-chat": "/share/pi/ema2016/models/meta-llama/Llama-2-70b-chat-hf",
    "llama-7b-chat": "/share/pi/ema2016/models/meta-llama/Llama-2-7b-chat-hf",
    "llama-7b-base": "/share/pi/ema2016/models/meta-llama/Llama-2-7b-hf",
    "llama-13b-base": "/share/pi/ema2016/models/meta-llama/Llama-2-13b-hf",
    "mellama-13b-chat": "/share/pi/ema2016/models/me-llama/MeLLaMA-13B-chat",
    "mellama-13b-base": "/share/pi/ema2016/models/me-llama/MeLLaMA-13B", 
    "mellama-70b-chat": "/share/pi/ema2016/models/me-llama/MeLLaMA-70B-chat",    
    "qwen3-30b": "/share/pi/ema2016/models/Qwen3-30B-A3B-Instruct-2507", 
}

LOCAL_RESULTS_DIR = "/share/pi/ema2016/users/sronaghi/proxy_tuning/results/medhelm"
# helpers adapted from unite 

def update_vocab(v1, vu, tokenizer, logits, model_name):
    for vu_token, v1_token, logit_ele in zip(vu,v1,logits):
        v1_token_ids = []
        for item in v1_token.values():
            v1_token_ids.append(item[1])
        for token in vu_token:  
            if token not in v1_token.keys():
                if 'llama' in model_name.lower():
                    token = token.replace('Ġ','▁')        
                if token != '':
                    subtoken_id = tokenizer.convert_tokens_to_ids(token)
                    if subtoken_id != 0 and subtoken_id != None: #Mistral and Llama2 oov id 0
                        logit = logit_ele[subtoken_id]
                    else:
                        subtokens = tokenizer.tokenize(token)
                        for token_id in tokenizer.convert_tokens_to_ids(subtokens):
                            #if 'llama2' in model_name:
                            if 'llama' in model_name.lower():
                                if token_id != 29871:
                                    subtoken_id = token_id
                                    break
                            else:
                                subtoken_id = token_id
                                break
                        logit = logit_ele[subtoken_id]
                else:
                    if 'qwen' in model_name.lower():
                        logit = logit_ele[220]
                        subtoken_id = 220
                    if 'llama' in model_name.lower():
                        logit = logit_ele[29871]
                        subtoken_id = 29871

                if 'llama' in model_name.lower():
                    v1_token[token.replace('▁', 'Ġ')] = [logit, subtoken_id]
                else:
                    if subtoken_id not in v1_token_ids:
                        v1_token[token] = [logit, subtoken_id]
                        v1_token_ids.append(subtoken_id)
                    else:
                        v1_token[token] = [0, subtoken_id] 
    v1_new = v1
    return v1_new

def vocab_softmax(v1):
        v1_new = []
        for element in v1:
            ele = {}
            ele_values = list(element.values())
            ele_values0, ele_values1 = [], []
            for item in ele_values:
                ele_values0.append(item[0])
                ele_values1.append(item[1])
            ele_values0 = torch.softmax(torch.tensor(ele_values0), dim=0)
            for token, prob, ids in zip(element.keys(),ele_values0,ele_values1):
                ele[token] = [prob, ids]
            v1_new.append(ele)

        return v1_new
    
    
def get_union_vocab(v1, v2):
    # Extract unique tokens from both dictionaries
        unique_tokens = []
        for v1_tokens, v2_tokens in zip(v1,v2):
            unique_tokens.append(list(set(v1_tokens.keys()) | set(v2_tokens.keys())))
        return unique_tokens
    

def get_top_k_tokens(logits, tokenizer, k=10):
    probs = logits

    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()
    top_k_probs = []
    for idx, prob in zip(top_k_indices,probs):
        prob_item = []
        for i in idx:
            prob_item.append(prob[i])
        top_k_probs.append(prob_item)

    top_k_tokens = []
    for indices in top_k_indices:
        token_item = []
        for idx in indices:
            token_item.append(tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True))
        top_k_tokens.append(token_item)

    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append(
            {token.replace('▁','Ġ').replace('<0x0A>','/n').replace('Ċ','/n'): [prob, int(id)] for token, prob, id in zip(token, prob, id)})

    return v1

# unite logit probability arithmetic
    
def unite_add(v1, v2, lamda, tokenizer):
    next_token_id1, next_token_id2 = [], []
    for element_v1, element_v2 in zip(v1, v2):
        assert len(element_v1) == len(element_v2)
        v_new = {}
        for token1 in element_v1:
            v_new[token1] = [lamda * element_v1[token1][0] + (1 - lamda) * element_v2[token1][0],
                             element_v1[token1][1]]
        probs = []
        for item in v_new.values():
            probs.append(item[0])
        sample_index = probs.index(max(probs))
        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
            i+=1
    return next_token_id1, next_token_id2


def capt_add(v1, v2, v3, alpha):
    next_token_id1, next_token_id2, next_token_id3 = [], [], []
    base_lp_chosen = dexpert_lp_chosen = None

    for element_v1, element_v2, element_v3 in zip(v1, v2, v3):

        v_new = {}
        
        for token1 in element_v1:
            v_new[token1] = [
                element_v1[token1][0] +
                (alpha * (element_v2[token1][0] - element_v3[token1][0])),
                element_v1[token1][1]
            ]

        probs = [item[0] for item in v_new.values()]
        sample_index = probs.index(max(probs))
        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
                next_token_id3.append(element_v3[item1][1])
                base_lp_chosen  = element_v1[item1][0]  
                dexpert_lp_chosen = v_new[item1][0] 
                if torch.is_tensor(dexpert_lp_chosen):
                    dexpert_lp_chosen = dexpert_lp_chosen.item()


            i += 1
            
    print("capt add is returning: " , next_token_id1, next_token_id2, next_token_id3, base_lp_chosen, dexpert_lp_chosen)
    return next_token_id1, next_token_id2, next_token_id3, base_lp_chosen, dexpert_lp_chosen


def add_pad_token(tok, padding_side="left"):
    # Ensure pad token exists and set padding side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok

class AnyModel:
    def __init__(
        self,
        base_name,
        expert_name,
        antiexpert_name,
        alpha: float = 1.0,
        unite: bool = False, 
        proxy: bool = False,
        model_kwargs: Dict[str, Any] = None
    ):
        
        self.expert = None  
        self.tok_exp = None
        self.antiexpert = None  
        self.tok_anti = None
        
        print("loading base model")
        
        if base_name in ["mellama-13b-chat", "mellama-13b-base"]:
            self.tok_base = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-base"], use_fast=True)     
        elif base_name in ["mellama-70b-chat"]:
            self.tok_base = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-chat"], use_fast=True)  
        else:
            self.tok_base = AutoTokenizer.from_pretrained(MODEL_PATHS[base_name], use_fast=True)   

        self.tok_base = add_pad_token(self.tok_base)

        print("done loading base tok", flush=True)
        
        self.base = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[base_name], **model_kwargs)
        self.base.eval()
      
        print("done loading base model")
        
        if proxy or unite:
            print("loading exp tok", flush=True)
            self.tok_exp = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-base"], use_fast=True)     
            self.tok_exp = add_pad_token(self.tok_exp)
            print("done loading exp tok")
            print("loading exp model")
            self.expert = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[expert_name], **model_kwargs)
            self.expert.eval()
            print("done loading exp model")
            
            if proxy:
                print("loading anti tok", flush=True)
                self.tok_anti = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-base"], use_fast=True) 
                self.tok_anti = add_pad_token(self.tok_anti)
                print("done loading anti tok", flush=True)
    
                print("loading anti model")
                self.antiexpert = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[antiexpert_name], **model_kwargs)
                self.antiexpert.eval()
                print("done loading anti model")
        
        self.alpha = alpha
        self.device = self.base.device

    
    def _encode_for_gen(self, tok, prompt: str, device=None, no_chat=False):
        text = prompt
        if not no_chat and getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        enc = tok(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", (input_ids != tok.pad_token_id).long())
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask, text

    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        if getattr(outputs, "cache_position", None) is not None:
        # some models already return it
            kwargs["cache_position"] = outputs.cache_position
        else:
            if "cache_position" in kwargs:
                kwargs["cache_position"] = kwargs["cache_position"] + 1
            else:
                # first step: position is sequence-length-1
                seq_len = kwargs["attention_mask"].shape[1]
                kwargs["cache_position"] = torch.arange(seq_len - 1, seq_len, device=kwargs["attention_mask"].device)

        return kwargs
    
    @torch.inference_mode() 
    def generate(
        self,
        prompt,
        max_new_tokens: Optional[int] = 700,
        alpha: float = 1.0,
        return_logits_for_analysis: bool = False,
        score_type=None,
        k=20,
        unite: bool = False,
        proxy: bool = False,
        **kwargs
    ):
        logit_results = None
        base_input_ids, base_attn, text = self._encode_for_gen(self.tok_base, prompt, device=self.base.device)
        print("prompt with (potential) instruction tag: ", text)
        base_kwargs = kwargs.copy()
        base_kwargs["attention_mask"] = base_attn
        base_kwargs["use_cache"] = True
        original_prompt_len = base_input_ids.shape[1]
        
        # if not proxy or unite, do generation using huggingface's generate function -- 
        if not proxy and not unite:
            gen = self.base.generate(
                input_ids=base_input_ids,
                attention_mask=base_attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tok_base.eos_token_id,
                pad_token_id=self.tok_base.pad_token_id,
            )
            gen_ids = gen[0, original_prompt_len:]
            generation = self.tok_base.decode(gen_ids, skip_special_tokens=True)
            return generation, logit_results
        
      
        if proxy or unite:
            expert_input_ids, expert_attn, expert_text = self._encode_for_gen(self.tok_exp, prompt, device=self.expert.device)
            expert_kwargs = kwargs.copy()
            expert_kwargs["attention_mask"] = expert_attn
            expert_kwargs["use_cache"] = False
            original_prompt_len_expert = expert_input_ids.shape[1]
            expert_prompt_ids = expert_input_ids[0, :original_prompt_len_expert]
            expert_prompt_decoded = self.tok_exp.decode(expert_prompt_ids, skip_special_tokens=True)
            if proxy:
                antiexpert_input_ids, anti_attn, anto = self._encode_for_gen(self.tok_anti, prompt, device=self.antiexpert.device)
                antiexpert_kwargs = kwargs.copy()
                antiexpert_kwargs["attention_mask"] = anti_attn
                antiexpert_kwargs["use_cache"] = False
                original_prompt_len_antiexpert = antiexpert_input_ids.shape[1]
                antiexpert_prompt_ids = antiexpert_input_ids[0, :original_prompt_len_antiexpert]
                antiexpert_prompt_decoded = self.tok_anti.decode(antiexpert_prompt_ids, skip_special_tokens=True)
                
        if proxy and score_type == "logits":
            expert_kwargs["use_cache"] = True
            antiexpert_kwargs["use_cache"] = True

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(1, dtype=torch.long, device=base_input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tok_base.eos_token_id], device=base_input_ids.device)
        
        if return_logits_for_analysis:
            T = max_new_tokens
            tok_ids     = torch.empty(T, dtype=torch.int32,  device="cpu")
            base_tok_ids = torch.empty(T, dtype=torch.int32,  device="cpu")
            base_lps    = torch.empty(T, dtype=torch.float32, device="cpu")
            dexpert_lps = torch.empty(T, dtype=torch.float32, device="cpu")
            argdiffs    = torch.empty(T, dtype=torch.int8,    device="cpu")  # 0 or 1
            t = 0


        for step in range(max_new_tokens): 
            if step == max_new_tokens - 1:
                print("hit max tokens")
            base_inputs = self.base.prepare_inputs_for_generation(base_input_ids, **base_kwargs)
            base_outputs = self.base(**base_inputs, return_dict=True)
            base_next_token_logits = base_outputs.logits[..., -1, :]
                
            next_token_id1 = next_token_id2 = next_token_id3 = None
            
            if not unite and not proxy:
                next_tokens = torch.argmax(base_next_token_logits, dim=-1)  # indices of top tokens
                next_token_id1 = next_tokens.tolist()              
            
            if proxy or unite:
                expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs) 
                expert_outputs = self.expert(**expert_inputs, return_dict=True)
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                
                if unite and prefix_allowed_tokens_fn_exp:
                    mask = torch.full_like(expert_next_token_logits, -math.inf) 
                    sent = expert_input_ids[0]
                    allowed = prefix_allowed_tokens_fn_exp(0, sent)
                    if len(allowed) == 0:
                        raise ValueError("prefix_allowed_tokens_fn returned an empty list.")
                    mask[0, allowed] = 0
                    expert_next_token_logits = expert_next_token_logits + mask
                    
                if unite:
                    v_base = get_top_k_tokens(base_next_token_logits, self.tok_base, k=k)
                    v_exp = get_top_k_tokens(expert_next_token_logits, self.tok_exp, k=k)
                    vu = get_union_vocab(v_base, v_exp)
                    v_base = update_vocab(v_base, vu, self.tok_base, base_next_token_logits,'qwen')
                    v_base = vocab_softmax(v_base)
                    v_exp = update_vocab(v_exp, vu, self.tok_exp, expert_next_token_logits,'llama')
                    v_exp = vocab_softmax(v_exp)

                    next_token_id1, next_token_id2 = unite_add(v_base,v_exp, 0.5, self.tok_base)

                elif proxy:
                    antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(antiexpert_input_ids, **antiexpert_kwargs) 
                    antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=True)
                    antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :] 

                    if score_type == "logprobs": #capt
                        base_lp = F.log_softmax(base_next_token_logits,  dim=-1)
                        expert_lp = F.log_softmax(expert_next_token_logits, dim=-1)
                        antiexpert_lp = F.log_softmax(antiexpert_next_token_logits, dim=-1)

                        v_base = get_top_k_tokens(base_lp, self.tok_base, k=k)
                        v_exp = get_top_k_tokens(expert_lp, self.tok_exp, k=0)
                        v_exp = update_vocab(v_exp, v_base, self.tok_exp, expert_lp,'llama')
                        v_anti = get_top_k_tokens(antiexpert_lp, self.tok_anti, k=0)
                        v_anti = update_vocab(v_anti, v_base, self.tok_anti, antiexpert_lp, 'llama')

                        next_token_id1, next_token_id2, next_token_id3, base_lp_chosen, dexpert_lp_chosen = capt_add(v_base, v_exp, v_anti, alpha)
                        if return_logits_for_analysis:
                            base_argmax_id = int(torch.argmax(base_lp[0]).item())
                            capt_choice_id = int(next_token_id1[0])
                            tok_ids[t]     = capt_choice_id
                            base_tok_ids[t]  = base_argmax_id
                            base_lps[t]    = float(base_lp_chosen)
                            dexpert_lps[t] = float(dexpert_lp_chosen)  
                            argdiffs[t]    = 1 if base_argmax_id != capt_choice_id else 0
                            t += 1
                           
                    elif score_type == "logits":  # regular proxy tuning 
                        expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]
                        next_token_logits = (
                            base_next_token_logits +
                            self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
                        )
                        next_tokens = torch.argmax(next_token_logits, dim=-1)  # indices of top tokens
                        next_token_id1 = next_tokens.tolist()
                        next_token_id2 = list(next_token_id1)
                        next_token_id3 = list(next_token_id1)
            
                        exp_step_ids = torch.as_tensor(next_token_id2,  device=expert_input_ids.device,   dtype=torch.long)
                        expert_input_ids = torch.cat([expert_input_ids,     exp_step_ids[:,  None]], dim=-1)
                        expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
                        anti_step_ids = torch.as_tensor(next_token_id3, device=antiexpert_input_ids.device, dtype=torch.long)
                        antiexpert_input_ids = torch.cat([antiexpert_input_ids, anti_step_ids[:, None]], dim=-1)
                        antiexpert_kwargs= self._update_model_kwargs_for_generation(antiexpert_outputs,antiexpert_kwargs)

            step_ids = torch.as_tensor(next_token_id1, device=base_input_ids.device, dtype=torch.long)
            base_input_ids = torch.cat([base_input_ids, step_ids[:, None]], dim=-1)
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            
            if (proxy and score_type == "logprobs") or unite:
                base_gen_ids = base_input_ids[0, original_prompt_len:]
                base_gen_decoded = self.tok_base.decode(base_gen_ids, skip_special_tokens=True)
                expert_input_decoded = expert_prompt_decoded + base_gen_decoded
                expert_input_ids, expert_kwargs["attention_mask"], expert_text = self._encode_for_gen(self.tok_exp, expert_input_decoded, device=self.expert.device, no_chat=True)
                if proxy:              
                    antiexpert_input_decoded = antiexpert_prompt_decoded + base_gen_decoded
                    antiexpert_input_ids, antiexpert_kwargs["attention_mask"], antiexpert_text = self._encode_for_gen(self.tok_anti, antiexpert_input_decoded, device=self.antiexpert.device,  no_chat=True)
                    
                if step < 15:
                    print(f"\n=== Step {step} ===")
                    print(f"Base decoded: {self.tok_base.decode(base_input_ids[0], skip_special_tokens=False)}")
                    if proxy or unite:
                        print(f"Expert decoded: {self.tok_exp.decode(expert_input_ids[0], skip_special_tokens=False)}")
                    if proxy:
                        print(f"Anti-expert decoded: {self.tok_anti.decode(antiexpert_input_ids[0], skip_special_tokens=False)}")
                    print(f"---")
                


            at_eos = (step_ids == eos_token_id_tensor[0]).long()
            unfinished_sequences = unfinished_sequences * (1 - at_eos)
            if unfinished_sequences.max() == 0:
                break

        gen_ids = base_input_ids[0, original_prompt_len:]
        generation = self.tok_base.decode(gen_ids, skip_special_tokens=True)
        
        
        if proxy and return_logits_for_analysis:
            logit_results = {
                "token_ids":   tok_ids[:t],
                "base_tok_ids": base_tok_ids[:t], 
                "base_lp":     base_lps[:t],
                "dexpert_lp":  dexpert_lps[:t],
                "argdiff":     argdiffs[:t],
            }

        
        return generation, logit_results

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# proxy tuning helpers

def _safe_tag(model_name: str) -> str:
    # e.g. "proxy_tuning/llama70b_mellama13bchat" -> "proxy_tuning_llama70b_mellama13bchat"
    return model_name.replace("/", "_").replace(" ", "").replace(".", "").replace("-", "")

def setup_run_dirs(model_name: str, root=LOCAL_RESULTS_DIR):
    """
    Creates:
      <root>/<TAG>_<YYYYMMDD_HHMMSS>/
          ├─ <TAG>_<YYYYMMDD_HHMMSS>.csv
          └─ logits_analysis/
    Returns: (run_dir, csv_path, logits_dir)
    """
    ensure_dir(root)
    tag = _safe_tag(model_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{tag}_{stamp}")
    ensure_dir(run_dir)

    csv_name = f"{tag}_{stamp}.csv"
    csv_path = os.path.join(run_dir, csv_name)
    with open(csv_path, "w") as f:
        f.write("timestamp,request_id,model_name,prompt,output,logits_path\n")

    logits_dir = os.path.join(run_dir, "logits_analysis")
    ensure_dir(logits_dir)

    print(f"[TokenLog] created run dir: {run_dir}")
    print(f"[TokenLog] csv: {csv_path}")
    print(f"[TokenLog] logits dir: {logits_dir}")
    return run_dir, csv_path, logits_dir


def append_request_row(csv_path: str, request_id: str, model_name: str, prompt: str, output: str, logits_path: str | None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def esc(s: str) -> str:
        if s is None: return ""
        return s.replace("\n", "\\n").replace(",", "&#44;")
    with open(csv_path, "a") as f:
        f.write(f"{ts},{request_id},{esc(model_name)},{esc(prompt)},{esc(output)},{esc(logits_path or '')}\n")


class ProxyTuningClient(Client):
    """
    A HELM client that uses ProxyTuning for inference instead of directly calling the model.
    """

    def __init__(
        self,
        model_name: str = None,
    ):
        """
        Initializes the ProxyTuningClient.

        """
        self.run_dir, self.token_log_path, self.logits_dir = setup_run_dirs(model_name)
        self.model_name = model_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.req_seq = 0
        tag = model_name.split("/")[-1]
        self.return_logits_for_analysis = True


        parts = tag.split("_")
        base_name, expert_name, antiexpert_name, self.alpha, self.score_type, k_str  = (
            parts[0],
            parts[1],
            parts[2],
            float(parts[3]),
            parts[4],
            parts[5]
        )
        self.k = int(k_str)
        self.is_unite = False
        self.is_proxy = False
        if expert_name != "none":
            if antiexpert_name == "none":
                self.is_unite = True
            else:
                self.is_proxy = True
        
        
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",         
            bnb_4bit_compute_dtype=torch.bfloat16,  
        )

        model_kwargs = {
            'device_map': "auto",
            'dtype': torch.bfloat16,
            'quantization_config': bnb_cfg,
            'low_cpu_mem_usage': True,
            'trust_remote_code': True,
        }

        print ("creating any model", flush=True)
        self.any_model = AnyModel(
            base_name=base_name,
            expert_name=expert_name,
            antiexpert_name=antiexpert_name,
            alpha=self.alpha,
            proxy=self.is_proxy,
            unite=self.is_unite,
            model_kwargs=model_kwargs,
        )

        print(f"[Loader] Base   : {base_name}", flush=True)
        print(f"[Loader] Expert : {expert_name}", flush=True)
        print(f"[Loader] Anti   : {antiexpert_name}", flush=True)

    
    def make_request(self, request: Request) -> RequestResult:
    
        prompt_text = request.prompt
        max_new_tokens=750
        
        if request.max_tokens:
            max_new_tokens=request.max_tokens
            print("max_new_tokens: ", max_new_tokens)

        if request.messages:
            print(request.messages)
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")
           
        # progress = tqdm.tqdm(total=1, desc="Generating Completions")
        print("doing a generation", flush=True)
        
        generation, logit_results = self.any_model.generate(
            prompt = prompt_text,
            max_new_tokens = max_new_tokens,
            alpha = self.alpha, 
            return_logits_for_analysis = self.return_logits_for_analysis,
            score_type = self.score_type,
            k = self.k,
            unite = self.is_unite,
            proxy = self.is_proxy,
        )
        

        print("generation: ", generation, flush=True)
        
        self.req_seq += 1
        request_id = f"{self.run_id}_r{self.req_seq:04d}"

        logits_path = None
        if self.return_logits_for_analysis and logit_results:
            logits_path = os.path.join(self.logits_dir, f"logits_{request_id}.pt")
            torch.save(logit_results, logits_path)
            print(f"[Logits] wrote {logits_path}")

        append_request_row(
            csv_path=self.token_log_path,
            request_id=request_id,
            model_name=self.model_name,
            prompt=prompt_text,
            output=generation,
            logits_path=logits_path,
        )
        
        # Return a HELM-compatible RequestResult
        output = GeneratedOutput(text=generation, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])
