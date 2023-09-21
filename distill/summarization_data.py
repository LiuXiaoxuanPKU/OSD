from typing import Tuple
import torch
import datasets
from transformers import T5Tokenizer

import json
import os
import re


# Wikihow Dataset

class Wikihow_Dataset(torch.utils.data.Dataset):
  
  def __init__(self, path, split, source_length, target_length) -> None:
      super().__init__()

      self.dataset = datasets.load_dataset("wikihow", "all", data_dir=path, split=split)
      self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
      self.source_length = source_length
      self.target_length = target_length

  def __len__(self) -> int:
    return  self.dataset.shape[0]
  
  def process_example(self, example):
    source = example['text']
    target = example['headline']

    source_tuple = self.tokenizer.encode(source, padding= 'max_length', max_length= self.source_length, truncation = True , return_tensors = "pt")
    print(source_tuple)
    target_tuple = self.tokenizer.encode(target, padding= 'max_length', max_length = self.target_length, truncation = True, return_tensors = "pt")

    return (source_tuple, target_tuple)

  def __getitem__(self, index) -> Tuple:

    example = self.dataset[index]
    source_tuple, target_tuple = self.process_example(example)

    source = source_tuple["input_ids"]
    source_mask = source_tuple["attention_mask"]

    target = target_tuple["input_ids"]
    target_mask = target_tuple["attention_mask"]

    return {"source": source, "source_mask": source_mask, "target": target, "target_mask": target_mask}


# Xsum Dataset

class Xsum_Dataset(torch.utils.data.Dataset):
  
  def __init__(self, split, source_length, target_length) -> None:
      super().__init__()

      self.dataset = datasets.load_dataset("xsum", split=split)
      self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
      self.source_length = source_length
      self.target_length = target_length

  def __len__(self) -> int:
    return  self.dataset.shape[0]
  
  def process_example(self, example):
    source = example['document']
    target = example['summary']

    source_tuple = self.tokenizer.batch_encode_plus([source,], padding= 'max_length', max_length= self.source_length, truncation = True , return_tensors = "pt")
    target_tuple = self.tokenizer.batch_encode_plus([target,], padding= 'max_length', max_length = self.target_length, truncation = True, return_tensors = "pt")

    return (source_tuple, target_tuple)

  def __getitem__(self, index) -> Tuple:

    example = self.dataset[index]
    source_tuple, target_tuple = self.process_example(example)

    source = source_tuple["input_ids"]
    source_mask = source_tuple["attention_mask"]

    target = target_tuple["input_ids"]
    target_mask = target_tuple["attention_mask"]

    return {"source": source, "source_mask": source_mask, "target": target, "target_mask": target_mask}
  

# summarization helper function
def preprocess_function_generic(examples, tokenizer, args, prefix=""):
  # remove pairs where at least one record is None

  # Get the column names for input/target.
  keys = list(examples.keys())
  #print(f'keys: {keys}')
  text_column = keys[0]
  summary_column = keys[1]

  # Temporarily set max_target_length for training.
  max_target_length = args.train_target_max_length
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
  model_inputs = tokenizer(inputs, max_length=args.source_max_length, padding=padding, truncation=True, return_tensors="pt", )

  # Tokenize targets with the `text_target` keyword argument
  labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt", )

  # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
  # padding in the loss.
  if padding == "max_length" and args.ignore_pad_token_for_loss:
      labels["input_ids"] = [
          [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
      ]

  model_inputs["labels"] = labels["input_ids"]
  model_inputs["decoder_attention_mask"] = labels["attention_mask"]
  return model_inputs