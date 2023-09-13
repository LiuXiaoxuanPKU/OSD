from datasets import load_dataset
import json
import random
import os

data_name = "ArtifactAI/arxiv-math-instruct-50k"
dataset = load_dataset(data_name)

# Access different splits
dataset["train"].to_json(f"math_train_raw.json")

def load_transform(filename, prefix):
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        if prefix == "train":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case['question']
                },
                {
                    "role" : "assistant",
                    "content" : case['answer']
                }
            ]
        elif prefix == "eval":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" :  case['question']
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

train_cases = load_transform(f"math_train_raw.json", "train")
# sample 200 cases only
random.shuffle(train_cases)
eval_cases = train_cases[:200]
train_cases = train_cases[200:]
print(len(train_cases), len(eval_cases))

with open(f'arxiv-math_train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open(f'arxiv-math_eval.json', 'w') as f:
    json.dump(eval_cases, f)

os.remove(f"math_train_raw.json")
