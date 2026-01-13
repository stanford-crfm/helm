import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def probe(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    print("\n==", name, "==")
    print("param dtype/device:", next(model.parameters()).dtype, next(model.parameters()).device)
    print("use_cache:", model.config.use_cache)
    print("attn_impl:", getattr(model.config, "_attn_implementation", None))
    print("device_map has cpu?:", any(v == "cpu" for v in (getattr(model, "hf_device_map", {}) or {}).values()))
    # prompt length
    messages = [{"role":"user","content":"Give me a short introduction to large language model."}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(model.device)
    print("prompt_tokens:", enc["input_ids"].shape[1])

    # short decode speed
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=32_000, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    gen = out.shape[1] - enc["input_ids"].shape[1]
    print("gen_tokens:", gen, "sec:", dt, "tok/s:", gen/dt)

probe("Qwen/Qwen3-4B-Thinking-2507")
probe("teetone/OpenR1-Distill-Qwen3-1.7B-Math")
