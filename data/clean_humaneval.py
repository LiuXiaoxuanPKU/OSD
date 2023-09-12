from datasets import load_dataset
import json
import random
import os
import pandas as pd

# Load a Parquet file into a Pandas DataFrame
dataset = load_dataset("openai_humaneval")
# Access different splits
dataset["test"].to_json("humaneval_raw.json")

def load_transform(filename):
    def transform(i, case):
        case["id"] = f"identity_{i}"
        case["conversation"] = [
            {
                "role" : "user",
                "content" : case["prompt"]
            },
            {
                "role" : "assistant",
                "content" : case["canonical_solution"]
            }
        ]
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

all_cases = load_transform("humaneval_raw.json")
print(len(all_cases))

with open('humaneval_eval.json', 'w') as f:
    json.dump(all_cases, f)

os.remove(f"humaneval_raw.json")