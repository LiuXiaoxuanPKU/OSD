import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from specInfer.generator import Generator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

teacher_model_path = "/data/starcoderbase"
student_model_path = "/data/starcoderbase-1b"
model = AutoModelForCausalLM.from_pretrained(teacher_model_path, 
                                         device_map='auto', 
                                         torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
small_model = AutoModelForCausalLM.from_pretrained(student_model_path, device_map='auto', torch_dtype=torch.bfloat16)

prompt = ("def print_hello_world():")
max_new_tokens = 100

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
# warmup
ref_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:]

###################################### Reference Generation ##############################
start = sychronize_time()
ref_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:]
print(f"Reference Time: {sychronize_time() - start}")
print(tokenizer.decode(ref_generated), end="\n\n")


###################################### Speculative Decoding ##############################
propose_num = 4
generator = Generator(small_model, model, tokenizer, propose_num)
start = sychronize_time()
print(inputs.input_ids.shape)
output = generator.generate(inputs.input_ids, 
                            max_new_tokens,
                            temperature=0.001)
print(f"Speculative Decoding Time: {sychronize_time() - start}")
print(f"alpha: {output.alpha_sum / output.sample_steps}, ",
      f"avg # of correct tokens: {output.correct_tokens.shape[-1] / output.propose_steps}")
print(output.output)