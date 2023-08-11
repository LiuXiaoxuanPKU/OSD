import json
import os
import argparse
import time
import torch
from fastchat.model import load_model

@torch.inference_mode()
def model_generate(model, input_ids, att_mask, past_key_values, position_ids=None):
    outputs = model(input_ids=input_ids, 
                        attention_mask=att_mask, 
                        past_key_values=past_key_values,
                        position_ids=position_ids)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    return next_token, tokenizer.batch_decode([next_token])[0], outputs.past_key_values   

def load_prompt(test_dir):
    test_file = os.path.join(test_dir, f"lines/testcases/680_lines.jsonl")
    with open(test_file, 'r') as f:
        line = f.readline()
    return json.loads(line)["prompt"]

def prepare_input(tokenizer, prompt_len, single_prompt):
    input_ids = tokenizer([single_prompt], return_tensors="pt").input_ids
    while input_ids.shape[-1] < prompt_len:
        input_ids = torch.cat([input_ids, input_ids], dim=-1)
    prompt = tokenizer.batch_decode(input_ids[:, :prompt_len])
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    return inputs

def bench_token_speed(prompt_len, decode_len, model, tokenizer, single_prompt):
    input = prepare_input(tokenizer, prompt_len, single_prompt)
    input_ids, attention_mask, past_key_values = input.input_ids, input.attention_mask, None
    token_times = [0] * decode_len
    repeat = 5
    for _ in range(repeat):
        for i in range(decode_len):
            start = time.time()
            next_token_id, _, past_key_values = model_generate(model, 
                                                                input_ids,
                                                                attention_mask, 
                                                                past_key_values)
            token_times[i] += time.time() - start
            input_ids = next_token_id.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long, device="cuda")], dim=-1)  
        
    return [t * 1.0 / repeat for t in token_times]  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, help="model path", default="/rscratch/zhendong/lily/longchat-7b-16k/")
    parser.add_argument("--test_dir", type=str, default="/home/eecs/zhen/lily/LongChat/longeval/evaluation", help="Directory of the testcases")
    
    args = parser.parse_args()
    from monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense
    replace_llama_with_condense()

    model, tokenizer = load_model(args.model_name_or_path,device="cuda")
    single_prompt = load_prompt(args.test_dir)
    
    # warmup
    bench_token_speed(4 * 1024, 1, model, tokenizer, single_prompt)
    for prompt_len in range(4, 6):
        prompt_len = prompt_len * 1024
        for decode_len in [1, 8, 64]:
            print(bench_token_speed(prompt_len, decode_len, model, tokenizer, single_prompt))
    
    # for prompt_len in range(32, 129, 32):
    #     prompt_len = prompt_len * 1024
    #     for decode_len in [1, 8, 64, 256, 1024]:
    #         bench_token_speed(prompt_len, decode_len, model, tokenizer, single_prompt)