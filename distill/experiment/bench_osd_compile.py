import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from specInfer.generator import Generator
from specInfer.common import timeit
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model_path = "/data/model/vicuna-7b-v1.3"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', use_fast=False)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
small_model = LlamaForCausalLM.from_pretrained("/data/model/llama-160m", torch_dtype=torch.bfloat16).to('cuda')

prompt = ("USER: Could you explain computer science to me? ASSISTANT:")
max_new_tokens = 100

inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
# warmup
ref_warmup_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
small_model = torch.compile(small_model, mode="reduce-overhead")
warmup_generated = small_model(**inputs)

# print("###################################### Reference Generation ##############################")
# out = timeit(model.generate, **inputs, max_new_tokens=max_new_tokens)
# print(tokenizer.batch_decode(out), end="\n\n")

print("###################################### Speculative Decoding ##############################")
def spec_decode(inputs, max_new_tokens):
    propose_num = 4
    with torch.no_grad():
        generator = Generator(small_model, model, tokenizer, propose_num, is_encoder_decoder=False)
        output = generator.generate(inputs.input_ids, max_new_tokens)
    return output
out = timeit(spec_decode, inputs, max_new_tokens)
print(out.output)