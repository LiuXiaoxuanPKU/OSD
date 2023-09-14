from collections import Counter
from datasets import load_dataset
import os
import json
import random

LIMIT = 10 * 1000  # we handle at most LIMIT data


class Collector:
    def __init__(self, name, *args, **kwargs) -> None:
        self.name = name
        self.dataset = load_dataset(self.name, *args, **kwargs)

    def get_raw_filename(self, split, prefix):
        if prefix is not None:
            return f"{prefix}_{self.name}_{split}_raw.json"
        else:
            return f"{self.name}_{split}_raw.json"

    def get_output_filename(self, split, prefix):
        if prefix is not None:
            return f"{prefix}_{self.name}_{split}_raw.json"
        else:
            return f"{self.name}_{split}.json"

    def collect(self, split, transform, size=None, prefix=None):
        rawfile = self.get_raw_filename(split, prefix)
        self.dataset[split].to_json(rawfile)
        with open(rawfile, "r") as f:
            raw_data = list(f)

            cases = []
            i = 0
            for case in raw_data:
                case = json.loads(case)
                cases.append(case)
                i += 1
                # if i % 1000 == 0:
                #     print(f"{i}/{LIMIT}")
                if i == LIMIT:
                    break
        cases = [transform(i, case) for i, case in enumerate(cases)]
        cases = [c for c in cases if (c is not None) and ('conversation' in c) ]
        case = {k: case[k] for k in ["id", "conversation"]}
        if size is not None:
            random.shuffle(cases)
            cases = cases[:size]

        os.remove(rawfile)
        with open(self.get_output_filename(split, prefix), 'w') as f:
            json.dump(cases, f)

        print(f"Dataset: {self.name}, Split: {split}, Length: {len(cases)}")
        raw_texts = [c["conversation"][0]["content"] for c in cases]
        self.count_unique_tokens(raw_texts)

    def count_unique_tokens(self, dataset):
        token_counter = Counter()
        for s in dataset:
            # we don't use tokenizer here because we just need a rough estimate
            tokens = s.split(' ')
            token_counter.update(tokens)
        unique_tokens = set(token_counter.keys())
        num_unique_tokens = len(unique_tokens)
        print(f"Number of unique tokens {num_unique_tokens}")
        return num_unique_tokens
