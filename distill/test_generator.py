from specInfer.generator import Generator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model_path = "/data/vicuna-13b-v1.5-16k"
model = LlamaForCausalLM.from_pretrained(
    model_path, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
small_model = LlamaForCausalLM.from_pretrained(
    "JackFram/llama-160m", device_map='auto', torch_dtype=torch.bfloat16)

prompt = ("USER: Could you explain computer science to me? ASSISTANT:")
max_new_tokens = 100

inputs = tokenizer([prompt], return_tensors="pt",
                   padding=True).to(model.device)
# warmup
ref_generated = model.generate(
    **inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:]

###################################### Reference Generation ##############################
start = sychronize_time()
ref_generated = model.generate(
    **inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:]
print(f"Reference Time: {sychronize_time() - start}")
print(tokenizer.decode(ref_generated), end="\n\n")


###################################### Speculative Decoding ##############################
propose_num = 4
generator = Generator(small_model, model, tokenizer, propose_num)
start = sychronize_time()
output = generator.generate(inputs.input_ids, max_new_tokens)
print(f"Speculative Decoding Time: {sychronize_time() - start}")
print(f"alpha: {output.alpha_sum / output.sample_steps},",
      f"correct tokens: {output.correct_tokens.shape[-1] / output.propose_steps}")
print(output.output)
