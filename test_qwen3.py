from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import time


model_name = "teetone/OpenR1-Distill-Qwen3-1.7B-Math"
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
start_time = time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32_000,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)
print("content:", content)
end_time = time.time()
print("inference time:", end_time - start_time)