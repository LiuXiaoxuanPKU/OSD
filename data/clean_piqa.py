from datasets import load_dataset
import json
import random

dataset = load_dataset("piqa")

# Access different splits
dataset["train"].to_json("piqa_train_raw.json")
dataset["test"].to_json("piqa_eval_raw.json")

def load_transform(filename, prefix):
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        if prefix == "train":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case["goal"]
                },
                {
                    "role" : "assistant",
                    "content" : case[f"sol{1 + int(case['label'])}"]
                }
            ]
        elif prefix == "eval":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case["goal"]
                }
            ]
        else:
            raise ValueError(prefix)
        case = {k: case[k] for k in ["id", "conversation"]}
        return case
    
    with open(filename, "r") as f:
        raw_data = list(f)

        cases = []
        for case in raw_data:
            case = json.loads(case)
            cases.append(case)
    cases = [transform(i, case) for i, case in enumerate(cases)]
    return cases

train_cases = load_transform("piqa_train_raw.json", "train")
eval_cases = load_transform("piqa_eval_raw.json", "eval")
# sample 200 cases only
random.shuffle(eval_cases)
eval_cases = eval_cases[:200]

with open('piqa_train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open('piqa_eval.json', 'w') as f:
    json.dump(eval_cases, f)

import os
os.remove(f"piqa_train_raw.json")
os.remove(f"piqa_eval_raw.json")