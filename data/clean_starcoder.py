from datasets import load_dataset
import json
from transformers import AutoTokenizer
import random
import os

# dataset = load_dataset("bigcode/starcoderdata", 
#                        data_dir="rust", 
#                        split="train")

# dataset.to_json(f"rustcode_train_raw.json")

def load_transform(filename, prefix):
    code_prompt = " Please only include Python code in your answer, don't include any explanation."
    tokenizer = AutoTokenizer.from_pretrained("/data/starcoderbase/")
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        input_ids = tokenizer(case['content'])["input_ids"]
        if len(input_ids) < 200:
            return None
        
        prompt = tokenizer.decode(input_ids[:200])
        label = tokenizer.decode(input_ids[200:])
    
        if prefix == "train":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" :  prompt
                },
                {
                    "role" : "assistant",
                    "content" : label
                }
            ]
        elif prefix == "eval":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : prompt
                }
            ]
        else:
            raise ValueError(prefix)
        case = {k: case[k] for k in ["id", "conversation"]}
        return case
    
    with open(filename, "r") as f:
        raw_data = list(f)

        i = 0
        cases = []
        for case in raw_data:
            case = json.loads(case)
            cases.append(case)
            i += 1
            if i == 20000:
                break
    cases = [transform(i, case) for i, case in enumerate(cases)]
    cases = [c for c in cases if c is not None]
    return cases


all_cases = load_transform(f"rustcode_train_raw.json", "train")
random.shuffle(all_cases)
eval_cases = all_cases[:200]
train_cases = all_cases[200:]
print(len(train_cases), len(eval_cases))

with open(f'rustcode_train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open(f'rustcode_eval.json', 'w') as f:
    json.dump(eval_cases, f)

os.remove(f"rustcode_train_raw.json")