from datasets import load_dataset
import json
import random
import os

data_name = "spider"
dataset = load_dataset(data_name)

# Access different splits
dataset["train"].to_json(f"{data_name}_train_raw.json")
dataset["validation"].to_json(f"{data_name}_eval_raw.json")

def load_transform(filename, prefix):
    SQL_prompt = " Please answer in SQL."
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        if prefix == "train":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case['question'] + SQL_prompt
                },
                {
                    "role" : "assistant",
                    "content" : " ".join(case['query_toks_no_value'])
                }
            ]
        elif prefix == "eval":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case['question'] + SQL_prompt
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

train_cases = load_transform(f"{data_name}_train_raw.json", "train")
eval_cases = load_transform(f"{data_name}_eval_raw.json", "eval")
# sample 200 cases only
random.shuffle(eval_cases)
eval_cases = eval_cases[:200]

with open(f'{data_name}_train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open(f'{data_name}_eval.json', 'w') as f:
    json.dump(eval_cases, f)

os.remove(f"{data_name}_train_raw.json")
os.remove(f"{data_name}_eval_raw.json")
