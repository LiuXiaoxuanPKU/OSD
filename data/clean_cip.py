from datasets import load_dataset
import json
import random

# dataset = load_dataset("alespalla/chatbot_instruction_prompts")

# # Access different splits
# dataset["train"].to_json("cip_train_raw.json")
# dataset["test"].to_json("cip_eval_raw.json")

def load_transform(filename, prefix):
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        if prefix == "train":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case["prompt"]
                },
                {
                    "role" : "assistant",
                    "content" : case["response"]
                }
            ]
        elif prefix == "eval":
            case["conversation"] = [
                {
                    "role" : "user",
                    "content" : case["prompt"]
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

train_cases = load_transform("cip_train_raw.json", "train")
eval_cases = load_transform("cip_eval_raw.json", "eval")
# sample 200 cases only
random.shuffle(eval_cases)
eval_cases = eval_cases[:200]

with open('cip_train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open('cip_eval.json', 'w') as f:
    json.dump(eval_cases, f)