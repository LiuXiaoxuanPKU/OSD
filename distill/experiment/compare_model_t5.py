import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import datasets
from functools import partial

import numpy as np

import argparse
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from specInfer.generator import Generator
from specInfer.generator_seq2seq import Seq2SeqGenerator
from train import LazySupervisedDataset

from distill.data import preprocess_function_spider

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, config=config).cuda()
    return model

def main(student_model_path,
         teacher_model_path,
         max_propose_num,
         dataset):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    tokenizer.pad_token = tokenizer.unk_token
    teacher_model = load_model(teacher_model_path)
    student_model = load_model(student_model_path)
    generator = Seq2SeqGenerator(student_model, teacher_model,
                          tokenizer, max_propose_num)

    raw_datasets = datasets.load_dataset(
            dataset,
    )
    partial_preprocess_function = partial(
        preprocess_function_spider,
        tokenizer=tokenizer,
        max_source_length=512, max_target_length=512
    )
    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.map(
        partial_preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Running tokenizer on eval dataset",
    )

    i = 0
    correctness = 0
    stats = torch.zeros(32000, dtype=torch.long, device='cuda')
    alpha, sample_steps = 0, 0
    for d in eval_dataset:
        max_tokens = 128

        input_ids = torch.Tensor(d["input_ids"]).long().reshape(1, -1).to(model.device)
        attention_mask = torch.Tensor(d["attention_mask"]).long().reshape(1, -1).to(model.device)

        output = generator.generate(input_ids, attention_mask, max_tokens, temperature=0.01)
        correct_tokens = output.correct_tokens.squeeze(0)
        print(5097 in correct_tokens.tolist())
        stats[correct_tokens] = stats[correct_tokens] + 1
        if i % 10 == 0:
            print(f"{i}/{len(eval_dataset)}")
        print("===================================")
        print('Question:')
        print(tokenizer.decode(d["input_ids"], skip_special_tokens=True))
        print('Answer:')
        print(output.output)
        print(correct_tokens.shape)
        print(output.propose_steps, correct_tokens.shape[-1]/output.propose_steps)
        print("===================================")
        correctness += output.correct_tokens.shape[-1]/output.propose_steps
        alpha += output.alpha_sum
        sample_steps += output.sample_steps
        i += 1
        if i == 5:
            break
    print(i, correctness / i, alpha / sample_steps)

def model_generate(model_path, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token
    model = load_model(model_path)
    param_sum = 0
    for param in model.parameters():
        param_sum += param.data.sum()
    print(f"Param Sum: {param_sum}")
    
    raw_datasets = datasets.load_dataset(
            dataset,
    )
    partial_preprocess_function = partial(
        preprocess_function_spider,
        tokenizer=tokenizer,
        max_source_length=1024, max_target_length=1024
    )
    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.map(
        partial_preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Running tokenizer on eval dataset",
    )

    i = 0
    for d in eval_dataset:
        max_tokens = 100
        input_ids = torch.Tensor(d["input_ids"]).long().reshape(1, -1).to(model.device)
        attention_mask = torch.Tensor(d["attention_mask"]).long().reshape(1, -1).to(model.device)
        generated = model.generate(input_ids, attention_mask,
                                   max_new_tokens=max_tokens, decoder_start_token_id=model.config.pad_token_id)[
            0]
        print(f"Prompt: {tokenizer.decode(input_ids[0])}")
        print(f"Answer: {tokenizer.decode(generated)}")
        print("----------------------------------")
        i += 1
        if i == 5:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str,
                        help="student model path",
                        default="google/t5-efficient-small")
    parser.add_argument("--teacher", type=str,
                        help="teacher model path",
                        default="google/t5-efficient-xl")
    parser.add_argument("--dataset", type=str,
                        help="data",
                        default="spider")
    parser.add_argument("--max_propose_num", type=int,
                        help="number of proposed tokens",
                        default=5)

    args = parser.parse_args()
    main(args.student, args.teacher, args.max_propose_num, args.dataset)
    model_generate(args.student, args.dataset)
