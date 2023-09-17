from typing import Tuple
import torch
import datasets
from transformers import T5Tokenizer

import json
import os
import re

# GSM8K Dataset
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/gsm8k/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

def preprocess_function_gsm8k(examples, tokenizer, args):
    if args.debug:
        inputs = examples["question"][:3]
        targets = examples["answer"][:3]
    else:
        inputs = examples["question"]
        targets = examples["answer"]
    padding = 'max_length'
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_source_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=args.max_target_length,
        padding=padding,
        truncation=True,
        return_tensors="pt", 
    )

    if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

def load_gsm8k(train_path: str = "./data/train.jsonl",
              test_path: str = "./data/test.jsonl"):
    train_data = datasets.Dataset.from_list(read_jsonl(train_path))
    test_data = datasets.Dataset.from_list(read_jsonl(test_path))
    return train_data, test_data

class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)

# SPIDER dataset
def preprocess_function_spider(examples, tokenizer, args, prefix="Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer."):
    if args.debug:
        inputs = examples["question"][:3]
        targets = examples["query"][:3]
    else:
        inputs = examples["question"]
        inputs = [prefix + inp for inp in inputs]
        targets = examples["query"]
    padding = 'max_length'
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_source_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=args.max_target_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )

    if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


    # WMT16 EN-DE dataset
def preprocess_function_ende(examples, tokenizer, args, prefix="translate English to German: "):
    if args.debug:
        example_text = examples['translation'][:2]
        inputs = example_text['en']
        targets = example_text["de"]
    else:
        all_text = examples['translation']
        
        inputs = []
        targets = []
        for excerpt in all_text:
            en_text = prefix + excerpt['en']
            de_text = excerpt['de']

            inputs.append(en_text)
            targets.append(de_text)
            
    padding = 'max_length'
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_source_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=args.max_target_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )

    if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs