from datasets import load_dataset
import json

dataset = load_dataset("alespalla/chatbot_instruction_prompts")

# Access different splits
dataset["train"].to_json("cip_train_raw.json")
dataset["test"].to_json("cip_eval_raw.json")

def load_transform(filename, prefix):
    def transform(i, case):
        case["id"] = f"{prefix}_identity_{i}"
        case["conversations"] = [
            {
                "from" : "human",
                "value" : case["prompt"]
            },
            {
                "from" : "gpt",
                "value" : case["response"]
            }
        ]
        case = {k: case[k] for k in ["id", "conversations"]}
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

with open('train.json', 'w') as f:
    json.dump(train_cases, f)
        
with open('eval.json', 'w') as f:
    json.dump(eval_cases, f)