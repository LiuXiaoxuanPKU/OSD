# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.model.model_adapter import get_conversation_template

from distill_trainer import DistillTrainer, DistillTrainerCallback

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    student_model_path: Optional[str] = field(
        default="facebook/opt-125m",  metadata={"help": "Path to student model"})
    teacher_model_path: str = field(
        default=None, metadata={"help": "Path to teacher model"})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_propose_num: int = field(
        default=5,
        metadata={
            "help": "gamma, number of tokens the student model proposes for each step"
        }
    )
    mode: str = field(
        default="offline",
        metadata={
            "help": "online mode or offline mode"
        }
    )
    online_eval_interval: int = field(
        default=10,
        metadata={
            "help": "evaluation interval for online training"
        }
    )
    online_update_interval: int = field(
        default=1,
        metadata={
            "help": "parameter update interval for online training"
        }
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    do_eval: bool
) -> Dict:
    if ("llama" in model.lower()) or ("starcoder" in model.lower()):
        # does not support multi-round conversation for llama
        # print(sources)
        if tokenizer.model_max_length > 1024 * 16:
            tokenizer.model_max_length = 1024 * 16
        if do_eval:
            assert len(sources) == 1
            conversations = [sources[0][0]["content"]]
            input_ids = tokenizer(
                conversations,
                return_tensors="pt"
            ).input_ids
        else:
            # Apply prompt templates
            conversations = []
            for i, source in enumerate(sources):
                conversations.append(
                    source[0]["content"] + source[1]["content"])
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
        targets = input_ids.clone()
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    elif "vicuna" in model.lower():
        conv = get_conversation_template(model)
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

        sources[0] = [
            {'role': 'user', 'content': sources[0][0]['content']},
            {'role': 'assistant', 'content': ''}
        ]

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["role"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["role"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["content"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        if do_eval:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt"
            ).input_ids
        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids

        labels = input_ids.clone()    
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    else:
        raise NotImplementedError(
            f"Does not support prompt template for {model}")


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model: str,
                 do_eval: bool = False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversation"]],
                         self.tokenizer,
                         self.model,
                         self.do_eval)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model: str
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert data_args.lazy_preprocess, "only support lazy process"
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json,
                                tokenizer=tokenizer,
                                model=model)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json,
                                   tokenizer=tokenizer,
                                   model=model,
                                   do_eval=True)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.student_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    # student model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.student_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    # # train from scratch
    # model = transformers.LlamaForCausalLM(config)

    # teacher model
    teacher_config = transformers.AutoConfig.from_pretrained(
        model_args.teacher_model_path,
        cache_dir=training_args.cache_dir,
    )
    if "starcoder" in model_args.teacher_model_path:
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_path,
            config=teacher_config,
            torch_dtype=torch.bfloat16
        )
    else:
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_path,
            config=teacher_config,
            torch_dtype=torch.bfloat16
        )
    teacher_model.cuda()
    print(
        f"Teacher Model memory: {teacher_model.get_memory_footprint() / 1024 / 1024} MB")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.teacher_model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              model=model_args.teacher_model_path)

    trainer = DistillTrainer(
        model=model, tokenizer=tokenizer,
        teacher_model=teacher_model,
        args=training_args, **data_module
    )
    trainer.add_callback(DistillTrainerCallback)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
