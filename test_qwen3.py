import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = r"""Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$"""

def probe(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    print("\n==", name, "==")
    print("param dtype/device:", next(model.parameters()).dtype, next(model.parameters()).device)
    print("use_cache:", model.config.use_cache)
    print("attn_impl:", getattr(model.config, "_attn_implementation", None))
    print("device_map has cpu?:", any(v == "cpu" for v in (getattr(model, "hf_device_map", {}) or {}).values()))

    # Build prompt/messages
    messages = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(model.device)
    print("prompt_tokens:", enc["input_ids"].shape[1])

    # Generate + time
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(
        **enc,
        max_new_tokens=32_000,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0

    sequences = out.sequences
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = sequences[0][prompt_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    gen = gen_ids.shape[0]
    print("gen_tokens:", gen, "sec:", dt, "tok/s:", gen / dt if dt > 0 else float("inf"))
    print("\n--- generated text ---")
    print(gen_text[-100:])

# gen_tokens: 1324 sec: 23.93214511871338 tok/s: 55.32308087855937
# gen_tokens: 1347 sec: 40.55997443199158 tok/s: 33.21008010640059
probe("Qwen/Qwen3-4B-Thinking-2507")
# gen_tokens: 32000 sec: 518.032811164856 tok/s: 61.77214900354351
probe("teetone/OpenR1-Distill-Qwen3-1.7B-Math")
