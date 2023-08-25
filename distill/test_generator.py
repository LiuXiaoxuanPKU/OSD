from specInfer.generator import Generator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model_path = "/data/longchat-7b-16k/"
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", device_map='auto', torch_dtype=torch.bfloat16)
    
prompt = ("what is your name? ")

inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
ref_generated = model.generate(**inputs, max_new_tokens=100)[0][inputs.input_ids.shape[-1]:]
start = sychronize_time()
ref_generated = model.generate(**inputs, max_new_tokens=100)[0][inputs.input_ids.shape[-1]:]
print(sychronize_time() - start)
print(tokenizer.decode(ref_generated), end="\n\n")


generator = Generator(small_model, model, tokenizer)
start = sychronize_time()
out, correct_ratio = generator.generate(inputs.input_ids, 200)
print(sychronize_time() - start)
print(f"{correct_ratio}: {out}")