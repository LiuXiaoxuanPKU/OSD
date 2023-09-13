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

import wandb

import random

import os

import datasets
import evaluate

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

# local import
from distill_trainer_seq2seq import Seq2SeqDistillTrainer, Seq2SeqDistillTrainerCallback
from summarization_data import Xsum_Dataset, Wikihow_Dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    student_model_path: Optional[str] = field(
        default="facebook/opt-125m",  metadata={"help": "Path to student model"})
    teacher_model_path: str = field(
        default=None, metadata={"help": "Path to teacher model"})


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "Dataset's name."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    metric: str = field(
        default='rouge', metadata={"help": "Metric to use for evaluation."}
    )
    source_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length for the source. Sequences will be right padded (and possibly truncated)."
        },
    )
    train_target_max_length: int = field(
        default=64,
        metadata={
            "help": "Maximum sequence length for the target during training. Sequences will be right padded."
        },
    )
    val_target_max_length: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length for the target during validation. Sequences will be right padded."
        },
    )
    test_target_max_length: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length for the target during testing. Sequences will be right padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_propose_num: int = field(
        default=5,
        metadata={
            "help": "gamma, number of tokens the student model proposes for each step."
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


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.student_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and data_args.source_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(data_args.source_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    # student model
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
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
    teacher_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.teacher_model_path,
        config=teacher_config
    )
    teacher_model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.teacher_model_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    # use only 1/8 training and testing data
    if data_args.dataset_name == 'xsum':
        train_dataset = Xsum_Dataset(split = 'train', source_length=data_args.source_max_length, target_length=data_args.train_target_max_length).dataset
        eval_dataset = Xsum_Dataset(split='validation', source_length=data_args.source_max_length, target_length=data_args.val_target_max_length).dataset
        predict_dataset = Xsum_Dataset(split='test', source_length=data_args.source_max_length, target_length=data_args.test_target_max_length).dataset
    elif data_args.dataset_name == 'wikihow':
        train_dataset = Wikihow_Dataset(split = 'train', path=os.path.join(os.getcwd(), 'data/wikihow/'), source_length=data_args.source_max_length, target_length=data_args.train_target_max_length)
        eval_dataset = Wikihow_Dataset(split='validation', path=os.path.join(os.getcwd(), 'data/wikihow/'), source_length=data_args.source_max_length, target_length=data_args.val_target_max_length)
        predict_dataset = Wikihow_Dataset(split='test', path=os.path.join(os.getcwd(), 'data/wikihow/'), source_length=data_args.source_max_length, target_length=data_args.test_target_max_length)
    else:
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]
        predict_dataset = raw_datasets["test"]

    # data partitioning
    train_random_indices = random.sample(range(len(train_dataset)), len(train_dataset)//8)
    train_dataset = datasets.Dataset.from_dict(train_dataset[train_random_indices])

    eval_random_indices = random.sample(range(len(eval_dataset)), 200)
    eval_dataset = datasets.Dataset.from_dict(eval_dataset[eval_random_indices])

    predict_random_indices = random.sample(range(len(predict_dataset)), len(predict_dataset)//8)
    predict_dataset = datasets.Dataset.from_dict(predict_dataset[predict_random_indices])

    # ---------------------------------------- Data preprocessing -----------------------------------------------
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.source_max_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.msource_max_length}."
            )
            model.resize_position_embeddings(data_args.msource_max_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.source_max_length)
        else:
            raise ValueError(
                f"`--source_max_length` is set to {data_args.source_max_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--source_max_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = train_dataset.column_names
    elif training_args.do_eval:
        column_names = eval_dataset.column_names
    elif training_args.do_predict:
        column_names = predict_dataset.column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    text_column = column_names[0]
    summary_column = column_names[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.train_target_max_length
    # default set padding to "max_length"
    padding = "max_length"

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.source_max_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_target_max_length
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.test_target_target_max_length
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on prediction dataset",
            )
    # ------------------------------------------------------------------------------------------------------------

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # -------------------------------------------------- Metric --------------------------------------------------
    metric = evaluate.load("rouge")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    # ------------------------------------------------------------------------------------------------------------
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    trainer = Seq2SeqDistillTrainer(
        model=model, 
        teacher_model=teacher_model, 
        tokenizer=tokenizer, 
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        propose_num=training_args.max_propose_num,
        args=training_args,
    )
    trainer.add_callback(Seq2SeqDistillTrainerCallback)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    model.config.use_cache = True
    trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
