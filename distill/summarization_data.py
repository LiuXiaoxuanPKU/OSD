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