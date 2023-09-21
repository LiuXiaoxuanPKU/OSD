import random
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from specInfer.common import sychronize_time, crop_past_key_values
from specInfer.generator import Generator


model_path = "/data/vicuna-7b-v1.3/"
# model_path = "JackFram/llama-160m"
model = LlamaForCausalLM.from_pretrained(
    model_path, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(
    "lmsys/vicuna-7b-v1.3")
tokenizer.padding_side = 'left'
small_model = LlamaForCausalLM.from_pretrained(
    "JackFram/llama-160m", device_map='auto', torch_dtype=torch.bfloat16)

torch.manual_seed(0)
prompt = ["USER: Could you explain computer science to me? ASSISTANT:",
          "USER: Hello? ASSISTANT:"]
input_ids = torch.randint(0, tokenizer.vocab_size, (2, 2048), device='cuda', dtype=torch.long)
input_ids = tokenizer(prompt, 
                      return_tensors='pt',
                      padding="max_length",
                      max_length=50,
                      truncation=True,
                      ).input_ids.cuda()
max_new_tokens = 100
# warmup
torch.manual_seed(0)
ref_generated = model.generate(
    input_ids=input_ids, max_new_tokens=max_new_tokens)[:, input_ids.shape[1]:]
print(tokenizer.batch_decode(ref_generated, skip_special_tokens=True), end="\n\n\n")

###################################### Reference Generation ##############################
torch.manual_seed(0)
start = sychronize_time()
ref_generated = model.generate(
    input_ids=input_ids, max_new_tokens=max_new_tokens)[:, input_ids.shape[1]:]
print(f"Reference Time: {sychronize_time() - start}")
print(tokenizer.batch_decode(ref_generated, skip_special_tokens=True), end="\n\n\n")


# ####################################### Mixed Sampling #####################################
def sample_token_from_logits(logits):
    tau = 0.001 # argmax
    distribution = torch.softmax(logits / tau, dim=-1)
    next_token_id = torch.multinomial(distribution, num_samples=1)
    return next_token_id


def get_mix_generated_ids(
    student_model,
    teacher_model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    mix_ratio
):
    def generate_one(model, input_ids, attention_mask, past_key_values):
        if past_key_values is None:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = outputs.past_key_values
        next_token = sample_token_from_logits(outputs.logits[:, -1, :])
        return next_token, past_key_values

    bsz, prompt_len = input_ids.shape
    # always generate the first token for teacher/student to get the kv cache
    student_first_token, student_key_values = generate_one(
        student_model, input_ids, attention_mask, None)
    teacher_first_token, teacher_key_values = generate_one(
        teacher_model, input_ids, attention_mask, None)
    
    torch.manual_seed(1)
    input_ids = student_first_token if random.random() < mix_ratio else teacher_first_token
    attention_mask = torch.cat([attention_mask, torch.ones(
            bsz, 1, dtype=torch.long, device='cuda')], dim=1)

    for i in range(max_new_tokens - 1):
        sample_model, past_key_values = (student_model, student_key_values) if random.random(
        ) < mix_ratio else (teacher_model, teacher_key_values)
        next_token, _ = generate_one(sample_model, input_ids, 
                                     attention_mask, past_key_values)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones(
            bsz, 1, dtype=torch.long, device='cuda')], dim=1)

    # mask eos
    print("before", (input_ids == tokenizer.pad_token_id).sum())
    eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, col in zip(*eos_positions):
        mask[row, col+1:] = True
    input_ids[mask] = tokenizer.pad_token_id
    print("after", (input_ids == tokenizer.pad_token_id).sum())
    return input_ids 
    


start = sychronize_time()
mix_ratio = 0.00001
generated_ids = get_mix_generated_ids(small_model, model, tokenizer,
                                      input_ids, torch.ones_like(input_ids),
                                      max_new_tokens, mix_ratio)
print(f"Mix Time: {sychronize_time() - start}")
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True), end="\n\n")
