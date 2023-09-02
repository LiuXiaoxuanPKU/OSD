from specInfer.generator import Generator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model_path = "/rscratch/zhendong/lily/longchat-7b-16k/"
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", device_map='auto', torch_dtype=torch.bfloat16)
    
prompt = (" A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Could you explain computer science to me? ASSISTANT:")

inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
ref_generated = model.generate(**inputs, max_new_tokens=100)[0][inputs.input_ids.shape[-1]:]
start = sychronize_time()
ref_generated = model.generate(**inputs, max_new_tokens=100)[0][inputs.input_ids.shape[-1]:]
print(sychronize_time() - start)
print(tokenizer.decode(ref_generated), end="\n\n")


propose_num = 5
generator = Generator(small_model, model, tokenizer, propose_num)
start = sychronize_time()
out, correct_tokens, propose_steps = generator.generate(inputs.input_ids, 200)
print(sychronize_time() - start)
print(f"{correct_tokens.shape[-1] / propose_steps}: {out}")