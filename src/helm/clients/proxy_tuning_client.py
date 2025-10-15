# File: helm/clients/proxy_tuning_client.py
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.cache import Cache
from helm.common.request import Request, RequestResult, GeneratedOutput

from typing import Optional, Dict, Any, List
import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
)
import tqdm
from transformers impor (
    BitsAndBytesConfig, 
)

from datetime import datetime

MODEL_PATHS = {
    "llama-70b-chat":  "[MODEL PATH]",
    "llama-13b-base": "[MODEL PATH]",
    "llama-7b-chat": "[MODEL PATH]",
    "mellama-13b-chat": "[MODEL PATH]",
    "mellama-13b-base": "[MODEL PATH]",
    "mellama-70b-chat": "[MODEL PATH]",    
    "qwen3-30b": "[MODEL PATH]", 
}

RESULTS_DIR = "[results dir]"
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
    
def average_and_sample(v1, v2, lamda, tokenizer):
    next_token, v_avg, next_token_id1, next_token_id2 = [], [], [], []
    for element_v1, element_v2 in zip(v1, v2):
        assert len(element_v1) == len(element_v2)
        v_new = {}
        for token1 in element_v1:
            v_new[token1] = [lamda * element_v1[token1][0] + (1 - lamda) * element_v2[token1][0],
                             element_v1[token1][1]]
        v_avg.append(v_new)
        probs = []
        for item in v_new.values():
            probs.append(item[0])
        sample_index = probs.index(max(probs))
        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
            i+=1
    return next_token, v_avg, next_token_id1, next_token_id2
    

def get_top_k_tokens(self, logits, tokenizer, k=10):
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

#proxy tuning approach
def logits_add(v1, v2, v3, tokenizer, alpha, device=None):
    next_token, next_token_id1, next_token_id2, next_token_id3 = [], [], [], []
    comb_ids_per_batch, comb_scores_per_batch = [], []

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
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
                next_token_id3.append(element_v3[item1][1])
            i += 1
        ids    = torch.tensor([v_new[t][1] for t in v_new], dtype=torch.long, device=device)
        scores = torch.tensor([v_new[t][0] for t in v_new], dtype=torch.float32, device=device)
        comb_ids_per_batch.append(ids)
        comb_scores_per_batch.append(scores)
    return next_token, next_token_id1, next_token_id2, next_token_id3, comb_ids_per_batch, comb_scores_per_batch



class DExpertsLlama:
    def __init__(
        self,
        base_name: str,
        expert_name: str,
        antiexpert_name: str,
        tokenizer_base, tokenizer_expert, tokenizer_anti,
        system_prompt: str = None,
        alpha: float = 1.0,
        unite: bool = False, 
        model_kwargs: Dict[str, Any] = None
    ):
        
        self.antiexpert = None  # ensure it exists
        self.tok_anti = None
        
        self.base = AutoModelForCausalLM.from_pretrained(
            base_name, **model_kwargs
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_name, **model_kwargs
        )
        self.base.eval()
        self.expert.eval()
       
        self.tok_base  = tokenizer_base
        self.tok_exp   = tokenizer_expert
        
        if not unite:
            self.antiexpert = AutoModelForCausalLM.from_pretrained(
                antiexpert_name, **model_kwargs
            )
            self.antiexpert.eval()
            self.tok_anti  = tokenizer_anti

        self.alpha = alpha
        self.device = self.base.device
        self.system_prompt = system_prompt
               

    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs=None,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        if antiexpert_inputs is not None:
            antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)
            return base_outputs, expert_outputs, antiexpert_outputs

        return base_outputs, expert_outputs
    

    def _get_chat_template_tokenized_chat_inputs(self, tokenizer, prompts):
        """
        Use tokenizer.apply_chat_template for models like Qwen-Instruct/Yi/Mistral-Instruct.
        Returns: input_ids (tensor on self.device)
        """
        def _msgs(p):
            if self.system_prompt:
                return [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": p}]
            return [{"role": "user", "content": p}]

        rendered = [
            tokenizer.apply_chat_template(_msgs(p), tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        chat_inputs = tokenizer(rendered, padding="longest", return_tensors="pt", add_special_tokens=True)
        return chat_inputs.input_ids.to(self.device)

    def _encode_plain_inputs(self, tokenizer, prompts):
        """
        Plain (non-chat) encoding with the given tokenizer.
        Returns: input_ids (tensor on self.device)
        """
        enc = tokenizer(prompts, padding="longest", return_tensors="pt", add_special_tokens=True)
        return enc.input_ids.to(self.device)
    
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
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        alpha: float = 1.0,
        return_logits_for_analysis: bool = False,
        score_type=None,
        k=20,
        unite: bool = False,
        **kwargs
    ):
        base_kwargs = kwargs.copy()

        # Decode to strings once using base tokenizer
        prompts = self.tok_base.batch_decode(input_ids, skip_special_tokens=True)

        if hasattr(self.tok_base, "apply_chat_template") and getattr(self.tok_base, "chat_template", None):
            base_input_ids = self._get_chat_template_tokenized_chat_inputs(self.tok_base, prompts)
        else:
            base_input_ids = self._encode_plain_inputs(self.tok_base, prompts)

        base_kwargs["attention_mask"] = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

     
        expert_kwargs = kwargs.copy()
        expert_input_ids     = input_ids
        
        if hasattr(self.tok_exp, "apply_chat_template") and getattr(self.tok_exp, "chat_template", None):
            expert_input_ids = self._get_chat_template_tokenized_chat_inputs(self.tok_exp, prompts)
        else:
            expert_input_ids = self._encode_plain_inputs(self.tok_exp, prompts)
        
        expert_kwargs['attention_mask']     = torch.ones_like(expert_input_ids,     dtype=torch.long, device=expert_input_ids.device)
        

        if not unite:  
            antiexpert_kwargs = kwargs.copy()
            antiexpert_input_ids = input_ids

            if hasattr(self.tok_anti, "apply_chat_template") and getattr(self.tok_anti, "chat_template", None):
                antiexpert_input_ids = self._get_chat_template_tokenized_chat_inputs(self.tok_anti, prompts)
            else:
                antiexpert_input_ids = self._encode_plain_inputs(self.tok_anti, prompts)
            antiexpert_kwargs['attention_mask'] = torch.ones_like(antiexpert_input_ids, dtype=torch.long, device=antiexpert_input_ids.device)
        
       
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tok_base.eos_token_id]).to(input_ids.device)
        
        T = max_new_tokens
        if (not unite) and return_logits_for_analysis:
            device = input_ids.device
            # 1 x T buffers on GPU
            p_dexperts = torch.empty(T, device=device, dtype=torch.bfloat16)
            p_base     = torch.empty(T, device=device, dtype=torch.bfloat16)
            p_expert   = torch.empty(T, device=device, dtype=torch.bfloat16)
            p_anti     = torch.empty(T, device=device, dtype=torch.bfloat16)

            preds_dexperts = torch.empty(T, device=device, dtype=torch.int32)
            preds_base     = torch.empty(T, device=device, dtype=torch.int32)
            preds_expert   = torch.empty(T, device=device, dtype=torch.int32)
            preds_anti     = torch.empty(T, device=device, dtype=torch.int32)

            token_ids_out  = torch.empty(T, device=device, dtype=torch.int32)
            t_write = 0

        for step in range(max_new_tokens):
            
            base_inputs = self.base.prepare_inputs_for_generation(base_input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
            
            
            if unite:
                base_outputs, expert_outputs = self.forward(
                    base_inputs, expert_inputs, return_dict=True
                )

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                v_base = self.get_top_k_tokens(base_next_token_logits, self.tok_base, k=k)
                v_exp = self.get_top_k_tokens(expert_next_token_logits, self.tok_exp, k=k)

                vu = get_union_vocab(v_base, v_exp)

                v_base = update_vocab(v_base, vu, self.tok_base, base_next_token_logits,'qwen')
                v_base = vocab_softmax(v_base)
                v_exp = update_vocab(v_exp, vu, self.tok_exp, expert_next_token_logits,'llama')
                v_exp = vocab_softmax(v_exp)

                next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(v_base,v_exp,0.5, self.tok_base)
            
            else:
                antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(antiexpert_input_ids, **antiexpert_kwargs)
                base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                    base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
                )

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
 
                if score_type == "logprobs":
                    base_next_token_logits  = F.log_softmax(base_outputs.logits[..., -1, :],  dim=-1)
                    expert_next_token_logits = F.log_softmax(expert_outputs.logits[..., -1, :], dim=-1)
                    antiexpert_next_token_logits = F.log_softmax(antiexpert_outputs.logits[..., -1, :], dim=-1)

                v_base = self.get_top_k_tokens(base_next_token_logits, self.tok_base, k=k)
                v_exp = self.get_top_k_tokens(expert_next_token_logits, self.tok_exp, k=0)
                v_exp = update_vocab(v_exp, v_base, self.tok_exp, expert_next_token_logits,'llama')
                v_anti = self.get_top_k_tokens(antiexpert_next_token_logits, self.tok_anti, k=0)
                v_anti = update_vocab(v_anti, v_base, self.tok_anti, antiexpert_next_token_logits, 'llama')

                next_token, next_token_id1, next_token_id2, next_token_id3, comb_ids, comb_scores = logit_add(v_base, v_exp, v_anti, self.tok_base, alpha, device=input_ids.device)
                
            next_tokens = torch.as_tensor(next_token_id1, device=input_ids.device, dtype=torch.long)
               
            input_ids      = torch.cat([input_ids,      next_tokens[:, None]], dim=-1)
            base_input_ids = torch.cat([base_input_ids, next_tokens[:, None]], dim=-1)

            exp_step_ids  = torch.as_tensor(next_token_id2,  device=expert_input_ids.device,   dtype=torch.long)
            expert_input_ids     = torch.cat([expert_input_ids,     exp_step_ids[:,  None]], dim=-1)
            
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            
            if not unite:
                anti_step_ids = torch.as_tensor(next_token_id3, device=antiexpert_input_ids.device, dtype=torch.long)
                antiexpert_input_ids = torch.cat([antiexpert_input_ids, anti_step_ids[:, None]], dim=-1)
                antiexpert_kwargs= self._update_model_kwargs_for_generation(antiexpert_outputs,antiexpert_kwargs)

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
                
        if (not unite) and return_logits_for_analysis:
            sl = slice(0, t_write)
            results = [{
                'token_ids':        token_ids_out[sl],     # [T’] int32 (GPU)
                'p_dexperts':       p_dexperts[sl],        # [T’] fp16  (GPU)
                'preds_dexperts':   preds_dexperts[sl],    # [T’] int32 (GPU)
                'p_base':           p_base[sl],
                'preds_base':       preds_base[sl],
                'p_expert':         p_expert[sl],
                'preds_expert':     preds_expert[sl],
                'p_antiexpert':     p_anti[sl],
                'preds_antiexpert': preds_anti[sl],
                # (optional) decode later if you want strings
            }]
            return input_ids, results
        return input_ids


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    add_special_tokens=True,
    disable_tqdm=False,
    return_logits_for_analysis=False,
    score_type=None,
    alpha=1.0,
    k=20,
    unite=False,
    **generation_kwargs, 
    
):
    generations = []
    outputs = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
     
    all_results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        
        # print ("tokenized_prompt: ", tokenized_prompts)
        if hasattr(model, "device"):                 # DExpertsLlama
            device = model.device
            # print ("device = model.device")
        else:                                        # vanilla HF model
            device = next(model.parameters()).device
            # print ("next(model.parameters()).devicedevice = next(model.parameters()).device")
        batch_input_ids = tokenized_prompts['input_ids'].to(device)
        attention_mask = tokenized_prompts['attention_mask'].to(device)
        
        batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                alpha=alpha,
                score_type=score_type,
                k=k,
                unite=unite,
                **generation_kwargs
        )
        results = []
        
        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(batch_input_ids, dict):
            batch_input_ids = batch_input_ids['llama']

        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        # print("batch_outputs: ", batch_outputs)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)
    # return generations, logits_for_analysis
    return generations, all_results


def add_pad_token(tokenizer, padding_side="left"):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer

def load_dexperts_model_and_tokenizer(
    base_name: str,
    expert_name: str,
    antiexpert_name: str,
    device_map: str = "auto",
    alpha: float = 1.0,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    system_prompt: Optional[str] = None,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
    proxy_tune: bool = False,
    unite: bool = False,
):
    
    bnb_cfg = None

    if load_in_8bit:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",         # {nf4, fp4}; nf4 is standard
            bnb_4bit_compute_dtype=torch.bfloat16,  
        )

    model_kwargs = {
        'device_map': device_map,
        'torch_dtype': torch.bfloat16,
        'quantization_config': bnb_cfg,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    }
    
    
    if "llama" in base_name and "chat" in base_name:
        tok_base = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-chat"], use_fast=use_fast_tokenizer)
    else:
        tok_base = AutoTokenizer.from_pretrained(MODEL_PATHS[base_name], use_fast=use_fast_tokenizer)
        
    if "llama" in expert_name and "chat" in expert_name:
        tok_exp = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-chat"], use_fast=use_fast_tokenizer)
    elif "llama" in expert_name and "chat" not in expert_name:
        tok_exp = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-13b-base"], use_fast=use_fast_tokenizer)
    else:
        tok_exp = AutoTokenizer.from_pretrained(MODEL_PATHS[expert_name], use_fast=use_fast_tokenizer)
        
    tok_base = add_pad_token(tok_base, padding_side)
    tok_exp  = add_pad_token(tok_exp,  padding_side)
    
    if proxy_tune:
        if "llama" in antiexpert_name and "chat" in antiexpert_name:
            tok_anti = AutoTokenizer.from_pretrained(MODEL_PATHS["llama-7b-chat"], use_fast=use_fast_tokenizer)
        else:
            tok_anti = AutoTokenizer.from_pretrained(MODEL_PATHS[antiexpert_name], use_fast=use_fast_tokenizer)

        tok_anti = add_pad_token(tok_anti, padding_side)
    
    
        model = DExpertsLlama(
                base_name=MODEL_PATHS[base_name],
                expert_name=MODEL_PATHS[expert_name],
                antiexpert_name=MODEL_PATHS[antiexpert_name],
                tokenizer_base=tok_base,
                tokenizer_expert=tok_exp,
                tokenizer_anti=tok_anti,
                system_prompt=system_prompt,
                alpha=alpha,
                model_kwargs=model_kwargs,
        )
        print(f"[Loader] Base   : {MODEL_PATHS[base_name]}")
        print(f"[Loader] Expert : {MODEL_PATHS[expert_name]}")
        print(f"[Loader] Anti   : {MODEL_PATHS[antiexpert_name]}")
        
    elif unite: 
        model = DExpertsLlama(
                base_name=MODEL_PATHS[base_name],
                expert_name=MODEL_PATHS[expert_name],
                antiexpert_name="none",
                tokenizer_base=tok_base,
                tokenizer_expert=tok_exp,
                tokenizer_anti="none",
                system_prompt=system_prompt,
                alpha=alpha,
                unite=True,
                model_kwargs=model_kwargs,
        )
        print(f"[Loader] Base   : {MODEL_PATHS[base_name]}")
        print(f"[Loader] Expert : {MODEL_PATHS[expert_name]}")
    
    return model, tok_base


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

        
def load_base_model_and_tokenizer(
    base_name: str,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    system_prompt: Optional[str] = None,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    bnb_cfg = None

    if load_in_8bit:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",         # {nf4, fp4}; nf4 is standard
            bnb_4bit_compute_dtype=torch.bfloat16,  
        )

    model_kwargs = {
        'device_map': device_map,
        'torch_dtype': torch.bfloat16,
        'quantization_config': bnb_cfg,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    }
    
    tok = AutoTokenizer.from_pretrained(MODEL_PATHS[base_name], use_fast=use_fast_tokenizer, trust_remote_code=True,)
    tok = add_pad_token(tok, padding_side)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[base_name],
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok

def get_chat_template_tokenized_chat_inputs(tokenizer, prompts):
    """
    Use tokenizer.apply_chat_template for models like Qwen/Yi/Mistral/Gemma-*.
    Returns a BatchEncoding dict with 'input_ids' and 'attention_mask'.
    """
    def _msgs(p):
        return [{"role": "user", "content": p}]

    # Render to string first, then tokenize → BatchEncoding (dict-like)
    rendered = [
        tokenizer.apply_chat_template(
            _msgs(p),
            tokenize=False,                # <-- important
            add_generation_prompt=True
        )
        for p in prompts
    ]
    enc = tokenizer(
        rendered,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    return enc

def base_generate_completions(
    model,
    tokenizer,
    prompts,
    max_new_tokens=600,
    do_sample=False,
):
    import torch
    model.eval()
    
    # if chat template 
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        enc = get_chat_template_tokenized_chat_inputs(tokenizer, prompts)
    else: 
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    # Ensure pad token is set
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Move to model device
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt portion
    prompt_len = enc["input_ids"].shape[1]
    new_tokens = gen_ids[:, prompt_len:]

    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    predicted_labels = decoded          
    all_results = None                  

    return  predicted_labels, all_results

    
    

class ProxyTuningClient(Client):
    """
    A HELM client that uses ProxyTuning for inference instead of directly calling the model.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        model_name: str = None,
        api_base: str = None,
        api_key: str = None,
    ):
        self.cache = Cache(cache_config)
        """
        Initializes the ProxyTuningClient.

        Args:
            tokenizer (Tokenizer): Tokenizer instance (unused but required by HELM interface).
            tokenizer_name (str): Name of the tokenizer (unused but required by HELM interface).
            cache_config (CacheConfig): Configuration for caching.

        """
        self.run_dir, self.token_log_path, self.logits_dir = setup_run_dirs(model_name)
        self.model_name = model_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.req_seq = 0
        tag = model_name.split("/")[-1]
        # strip optional "proxy_tuning_" prefix
        if tag.startswith("proxy_tuning_"):
            tag = tag[len("proxy_tuning_"):]

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

        print("mn:", model_name)
        print("tag:", tag)
        print("b: ", base_name)
        print("Ex:", expert_name)
        print("ax", antiexpert_name)
        print(self.alpha)
        print(self.score_type)
        print(self.k)
        print("proxy: ", self.is_proxy)
        print("unite: ", self.is_unite)
        
        if self.is_proxy: 
            self.model, self.hf_tokenizer = load_dexperts_model_and_tokenizer(
                    base_name=base_name,
                    expert_name=expert_name,
                    antiexpert_name=antiexpert_name,
                    load_in_8bit=False,
                    load_in_4bit=True,
                    use_fast_tokenizer=True,
                    system_prompt=None,
                    device_map='auto', 
                    proxy_tune=self.is_proxy
            )
        elif self.is_unite:
            self.model, self.hf_tokenizer = load_dexperts_model_and_tokenizer(
                    base_name=base_name,
                    expert_name=expert_name,
                    antiexpert_name=antiexpert_name,
                    load_in_8bit=False,
                    load_in_4bit=True,
                    use_fast_tokenizer=True,
                    system_prompt=None,
                    device_map='auto', 
                    proxy_tune=self.is_proxy, 
                    unite=self.is_unite
            )
            
        else:
            self.model, self.hf_tokenizer = load_base_model_and_tokenizer(
                base_name=base_name,
                load_in_4bit=False,
                device_map="auto",
                use_fast_tokenizer=True,
            )
    
    def make_request(self, request: Request) -> RequestResult:
        """
        Handles a request by sending the prompt 

        Args:
            request (Request): The request object containing the prompt.

        Returns:
            RequestResult: A HELM-compatible response object.
        """
        prompt_text = request.prompt

        if request.messages:
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")

        
        print("prompt_text: ", prompt_text)
        prompts = [prompt_text]
         # turn prompt into a [] 
        if self.is_proxy or self.is_unite: 
            predicted_labels, all_results = generate_completions(
                model=self.model,
                tokenizer=self.hf_tokenizer,
                prompts=prompts,
                max_new_tokens=600,       
                do_sample=False,        
                num_return_sequences=1,
                alpha=self.alpha,
                k=self.k,
                score_type=self.score_type,
                unite=self.is_unite,
                return_logits_for_analysis=False, 
            )
        else: 
            predicted_labels, all_results = base_generate_completions(
                model=self.model,
                tokenizer=self.hf_tokenizer,
                prompts=prompts,
                max_new_tokens=600,
                do_sample=False,
            )

            
        output_text = predicted_labels[0]
        print("output_text: ", output_text)
        
        self.req_seq += 1
        request_id = f"{self.run_id}_r{self.req_seq:04d}"

        logits_path = None
        if self.is_proxy and all_results:
            logits_path = os.path.join(self.logits_dir, f"logits_{request_id}.pt")
            torch.save(all_results, logits_path)
            print(f"[Logits] wrote {logits_path}")

        append_request_row(
            csv_path=self.token_log_path,
            request_id=request_id,
            model_name=self.model_name,
            prompt=prompt_text,
            output=output_text,
            logits_path=logits_path,
        )
        
        # Return a HELM-compatible RequestResult
        output = GeneratedOutput(text=output_text, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])
