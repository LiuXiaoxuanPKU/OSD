from datasets import load_dataset
import json
import random
import os

data_name = "mbpp"
dataset = load_dataset(data_name)

# Access different splits
dataset["train"].to_json(f"{data_name}_train_raw.json")
dataset["test"].to_json(f"{data_name}_eval_raw.json")
dataset["validation"].to_json(f"{data_name}_validation_raw.json")

def load_transform(filename, prefix):
    code_prompt = " Please only include Python code in your answer, don't include any explanation."
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        if prefix == "train":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" :  case['text'] + code_prompt
                },
                {
                    "role" : "assistant",
                    "content" : case['code']
                }
            ]
        elif prefix == "eval":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case['text'] + code_prompt
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

train1_cases = load_transform(f"{data_name}_train_raw.json", "train")
train2_cases = load_transform(f"{data_name}_eval_raw.json", "eval")
train_cases = train1_cases + train2_cases
eval_cases = load_transform(f"{data_name}_validation_raw.json", "eval")
# sample 200 cases only
random.shuffle(eval_cases)
eval_cases = eval_cases[:200]
print(len(train_cases), len(eval_cases))

with open(f'{data_name}_train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open(f'{data_name}_eval.json', 'w') as f:
    json.dump(eval_cases, f)

os.remove(f"{data_name}_train_raw.json")
os.remove(f"{data_name}_eval_raw.json")
