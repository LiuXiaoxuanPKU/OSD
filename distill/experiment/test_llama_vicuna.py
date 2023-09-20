import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from specInfer.generator import Generator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import random

model_path = "/data/vicuna-7b-v1.3"
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
small_model = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", device_map='auto', torch_dtype=torch.bfloat16)

prompt = ("USER: Could you explain computer science to me? ASSISTANT:")
prompts = [
     "USER: Could you explain computer science to me? ASSISTANT:",
     "USER: What's the result of 1+1? ASSISTANT:" 
]
max_new_tokens = 100

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
# warmup
ref_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

###################################### Reference Generation ##############################
start = sychronize_time()
ref_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
print(f"Reference Time: {sychronize_time() - start}")
print(tokenizer.batch_decode(ref_generated), end="\n\n")


# ###################################### Speculative Decoding ##############################
# propose_num = 4
# generator = Generator(small_model, model, tokenizer, propose_num)
# start = sychronize_time()
# output = generator.generate(inputs.input_ids, max_new_tokens)
# print(f"Speculative Decoding Time: {sychronize_time() - start}")
# print(f"alpha: {output.alpha_sum / output.sample_steps}, ",
#       f"avg # of correct tokens: {output.correct_tokens.shape[-1] / output.propose_steps}")
# print(output.output)

####################################### Mixed Sampling #####################################
def get_mix_generated_ids(
        student_model,
        teacher_model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        mix_ratio
    ):
        bsz = input_ids.shape[0]
        for i in range(max_new_tokens):
            sample_model = student_model if random.random() < mix_ratio else teacher_model
            outputs = sample_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_ids = outputs["sequences"]
            attention_mask = torch.cat([attention_mask, torch.ones(
                bsz, 1, dtype=torch.long, device="cuda")], dim=-1)
        return input_ids
  
start = sychronize_time()
mix_ratio = 0.000001
print(tokenizer('</s>'))
generated_ids = get_mix_generated_ids(small_model, model, tokenizer, 
                                      inputs.input_ids, inputs.attention_mask, 
                                      max_new_tokens, mix_ratio)
print(f"Mix Time: {sychronize_time() - start}")
print(tokenizer.batch_decode(generated_ids), end="\n\n")