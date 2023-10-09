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
from train import LazySupervisedDataset

def preprocess_function_spider(examples, tokenizer, max_source_length, max_target_length, prefix="translate the following question into SQL. Please only generate SQL, don't include explanation in the answer: "):
    inputs = examples["question"]
    inputs = [prefix + inp for inp in inputs]
    targets = examples["query"]
    padding = 'max_length'
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
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

# genric preprocess function, generally applicable to summarization tasks
def preprocess_function_generic(examples, tokenizer, max_source_length, max_target_length, prefix="Summarize the following excerpt: "):
  # remove pairs where at least one record is None

  # Get the column names for input/target.
  keys = list(examples.keys())
  #print(f'keys: {keys}')
  text_column = keys[0]
  summary_column = keys[1]

  # default set padding to "max_length"
  padding = "max_length"

  # remove pairs where at least one record is None
  inputs, targets = [], []
  for i in range(len(examples[text_column])):
      if examples[text_column][i] and examples[summary_column][i]:
          prompt = prefix + examples[text_column][i]
          inputs.append(prompt)
          targets.append(examples[summary_column][i])

  print(f'dataset length: {len(inputs)}')
  model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True, return_tensors="pt", )

  # Tokenize targets with the `text_target` keyword argument
  labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt", )

  # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
  # ignore padding in the loss.
  labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
  ]

  model_inputs["labels"] = labels["input_ids"]
  model_inputs["decoder_attention_mask"] = labels["attention_mask"]
  return model_inputs

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, config=config).cuda()
    return model

def main(student_model_path,
         teacher_model_path,
         max_propose_num,
         dataset,
         dataset_config_name=None):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    tokenizer.pad_token = tokenizer.unk_token
    teacher_model = load_model(teacher_model_path)
    student_model = load_model(student_model_path)
    generator = Generator(student_model, teacher_model,
                          tokenizer, max_propose_num, is_encoder_decoder=True)

    raw_datasets = datasets.load_dataset(
            dataset,
            dataset_config_name
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

    # take a subset
    test_suite_size = 5
    indices = range(len(eval_dataset))[-test_suite_size:]
    eval_dataset = eval_dataset.select(indices)

    correctness = 0
    stats = torch.zeros(32000, dtype=torch.long, device='cuda')
    alpha, sample_steps = 0, 0
    for d in eval_dataset:
        max_tokens = 128

        input_ids = torch.Tensor(d["input_ids"]).long().reshape(1, -1).to(student_model.device)
        attention_mask = torch.Tensor(d["attention_mask"]).long().reshape(1, -1).to(student_model.device)

        output = generator.generate(input_ids,max_tokens,  attention_mask=attention_mask, temperature=0.01)
        correct_tokens = output.correct_tokens.squeeze(0)
        print(5097 in correct_tokens.tolist())
        stats[correct_tokens] = stats[correct_tokens] + 1
        print("===================================")
        print('Question:')
        print(tokenizer.decode(d["input_ids"], skip_special_tokens=True))
        print('Answer:')
        print(output.output)
        print(correct_tokens.shape)
        print(output.propose_steps, correct_tokens.shape[-1]/output.propose_steps)
        print("===================================")
        correctness += output.correct_tokens.shape[-1]/output.propose_steps
        alpha += output.alpha_sum.item()
        sample_steps += output.sample_steps
    print(test_suite_size, correctness / test_suite_size, alpha / sample_steps)

def model_generate(model_path, dataset, dataset_config_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token
    model = load_model(model_path)
    param_sum = 0
    for param in model.parameters():
        param_sum += param.data.sum()
    print(f"Param Sum: {param_sum}")
    
    raw_datasets = datasets.load_dataset(
            dataset,
            dataset_config_name
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

    # take a subset
    test_suite_size = 5
    indices = range(len(eval_dataset))[-test_suite_size:]
    eval_dataset = eval_dataset.select(indices)

    for d in eval_dataset:
        max_tokens = 128
        input_ids = torch.Tensor(d["input_ids"]).long().reshape(1, -1).to(model.device)
        attention_mask = torch.Tensor(d["attention_mask"]).long().reshape(1, -1).to(model.device)
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_new_tokens=max_tokens, decoder_start_token_id=model.config.pad_token_id)[
            0]
        print(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        print(f"Answer: {tokenizer.decode(generated, skip_special_tokens=True)}")
        print("----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str,
                        help="student model path",
                        default="google/t5-efficient-small")
    parser.add_argument("--teacher", type=str,
                        help="teacher model path",
                        default="google/flan-t5-xl")
    parser.add_argument("--dataset", type=str,
                        help="data",
                        default="spider")
    parser.add_argument("--dataset_config_name", type=str,
                        help="confg name",
                        default=None)
    parser.add_argument("--max_propose_num", type=int,
                        help="number of proposed tokens",
                        default=5)

    args = parser.parse_args()
    main(args.student, args.teacher, args.max_propose_num, args.dataset)
    print('-----------------------------------------------------------')
    print('------------------ HF teacher generation ------------------')
    print('-----------------------------------------------------------')
    model_generate(args.teacher, args.dataset)
    print('-----------------------------------------------------------')
    print('------------------ HF student generation ------------------')
    print('-----------------------------------------------------------')
    model_generate(args.student, args.dataset)
