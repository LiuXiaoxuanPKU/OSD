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

import os

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer, BitsAndBytesConfig, PretrainedConfig

from fastchat.model.model_adapter import get_conversation_template

from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from student_predictor_osd import PredictorModel, PredictorOSDConfig

from operator import itemgetter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="models/vicuna-160m",  metadata={"help": "Path to student model"})
    teacher_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to teacher model"})
    predictor_head_name_or_path: str = field(
        default=None, metadata={"help": "Path to student model predictor head"})
    
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


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
        default=10,
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
    sample_source: str = field(
        default="student",
        metadata = {
            "choices" : ["student", "teacher", "mix_request", "mix_token"]
        }
    )
    kl_method: str = field(
        default="forward",
        metadata = {
            "choices" : ["forward", "reverse", "jsd"]
        }
    )
    predictor_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Predictor heads."},
    )
    predictor_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Predictor head."},
    )

# Customized for training Predictor heads
class PredictorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            predictor = model.module.predictor
        else:
            predictor = model.predictor

        # TODO: support both offline label mode (decouple teacher & student in a single pipeline)
        # current version: training mode with an online teacher model for verification
        teacher_model = self.model.teacher_model

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_lens = [torch.sum(attention_mask[i]).item() for i in range(input_ids.shape[0])]
        all_eos_flag = [False for _ in range(input_ids.shape[0])]
        
        critical_dim, critical_len = max(enumerate(prompt_lens), key=itemgetter(1))

        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
            
        iter_counter = 0
        while not all(all_eos_flag) and critical_len + iter_counter < self.args.model_max_length:
            #print(self.tokenizer.decode(input_ids[0]))
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            predictor_values, generated_seq = outputs[0][0], outputs[1]
            new_tokens = generated_seq[:, -1]

            with torch.inference_mode():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            teacher_next_token_logits = teacher_outputs.logits[:, -1, :]
            teacher_new_tokens = []
            for i in range(teacher_next_token_logits.shape[0]):
                teacher_new_token = torch.argmax(torch.softmax(teacher_next_token_logits[i, :] / 0.001, dim=-1), dim=-1)
                teacher_new_tokens.append(teacher_new_token)

            # inspect whether the teacher output contains eos
            # update the flag accordingly
            for b in range(input_ids.shape[0]):
                if all_eos_flag[b] == True:
                    continue

                new_token = new_tokens[b]
                if new_token == self.tokenizer.eos_token_id:
                    print(f'Seq: {b} hits eos. Training continuing as other seqs are still running.')
                    all_eos_flag[b] = True

            
            # create binary classification labels 
            labels = []
            for b in range(input_ids.shape[0]):
                student_token = new_tokens[b]
                teacher_token = teacher_new_tokens[b]
                if student_token == teacher_token:
                    # True for the binary classifier
                    labels.append([0,1])
                else:
                    # False
                    labels.append([1,0])
            
            # (bsz, num predictor heads, predictor value size)
            # TODO: support the multi-predictor head scenario
            predictor_labels = torch.Tensor(labels).to(predictor_values.device, dtype=float)
            #print(predictor_labels)

            new_predictor_labels = []
            #print(predictor_labels.shape)
            for i in range(predictor):           
                #print(predictor_values.shape)
                #print(predictor_labels.shape)

                # iterate over batch size dim, if no eos, compute loss
                loss_i = 0
                for b in range(input_ids.shape[0]):
                    if not all_eos_flag[b]:
                        #print(f'adding loss of seq: {b}')
                        seq_loss_i = loss_fct(predictor_values[b, ...], predictor_labels[b, ...])
                        loss_i += seq_loss_i
                loss += loss_i
                not_ignore = predictor_labels.ne(IGNORE_TOKEN_ID)
                predictor_labels = predictor_labels[not_ignore]
                new_predictor_labels.append(predictor_labels)
            
            # expand input_ids by a length of 1 along dim -1
            zeros = torch.zeros(input_ids.shape[0], 1).to(input_ids.device, dtype=int)
            input_ids = torch.cat((input_ids, zeros), dim=1)

            # add new token to left-padded sequence
            for b in range(input_ids.shape[0]):
                input_ids[b, -1] = teacher_new_tokens[b]
            
            attention_mask = generated_seq.ne(self.tokenizer.pad_token_id)

            iter_counter += 1

        print('iteration counter:')
        print(iter_counter)

        self.log(log)
        return (loss, new_predictor_values) if return_outputs else loss


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
    prompts,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    predictor_num_heads: int = 1,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        prompts: A list of prompts from chatbot arena dataset.
        sources: A list of speculative decoding-related record fields from chatbot arena dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """
    conv = get_conversation_template("vicuna")

    # Apply prompt templates
    conversations = []
    for i, prompt in enumerate(prompts):
        # remove bos
        prompt = prompt[4:]
        
        prompt = prompt.replace('\n', '')

        # prompt from dataset already in vicuna format
        conversations.append(prompt)

    all_predictor_labels = []
    for i, source in enumerate(sources):

        # each record_i entry
        for j, record_i in enumerate(source):
            labels = record_i['predictor_labels']

            if len(labels) < predictor_num_heads:
                raise ValueError(f'label length from the dataset should be larger than the number of heads requested: {predictor_num_heads}')

            for label in labels[:predictor_num_heads]:
                if label == 0:
                    all_predictor_labels += [int(1),int(0)]
                else:
                    all_predictor_labels += [int(0),int(1)]
    
    # hard-coded padding length to 8192: (8192/num_heads) max records (8192 max records when num_heads=1)
    # TODO: make this configurable
    if len(all_predictor_labels) < 8192:
        padding_length = 8192 - len(all_predictor_labels)
        all_predictor_labels += [-1] * padding_length
    else:
        all_predictor_labels = all_predictor_labels[:8192]
    
    #print(len(all_predictor_labels))

    # Tokenize conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=[all_predictor_labels],
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, predictor_num_heads, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        prompts = [example['prompt'] for prompt in raw_data]
        sources = [example["sd_records"] for example in raw_data]
        data_dict = preprocess(prompts, sources, tokenizer, predictor_num_heads)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, predictor_num_heads, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.predictor_num_heads = predictor_num_heads

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["prompt"]], [self.raw_data[i]["sd_records"]], self.tokenizer, self.predictor_num_heads)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, predictor_num_heads, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, predictor_num_heads, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, predictor_num_heads, tokenizer=tokenizer)
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
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load teacher model and tokenizer
    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.teacher_model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if model_args.load_in_4bit else None,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
    )

    # Load model and tokenizer
    # ignore quantization for now
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    # Freeze the base model
    for n, p in model.named_parameters():
        #print(n)
        p.requires_grad = False

    # Add Predictor heads
    predictor_lm_head = PredictorModel(
        model,
        teacher_model,
        predictor_num_heads=training_args.predictor_num_heads,
        predictor_num_layers=training_args.predictor_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}_predictor_mlp_{model_args.model_name_or_path.split('/')[-1]}_predictor_{training_args.predictor_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.predictor_num_layers}"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="left",
    )
    #tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, predictor_num_heads=training_args.predictor_num_heads, data_args=data_args)

    # Generate Predictor config for pushing to HF hub
    predictor_config = PredictorOSDConfig(
        predictor_num_heads=training_args.predictor_num_heads,
        predictor_num_layers=training_args.predictor_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    # Save Predictor config
    predictor_config.save_pretrained(training_args.output_dir)

    # import pdb; pdb.set_trace()
    # Start trainner
    trainer = PredictorTrainer(
        model=predictor_lm_head, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    # trainer.save_state()
    
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # Save PredictoraHead seperately
    if hasattr(predictor_lm_head, "module"):
        lm_head = predictor_lm_head.module.predictor_head
    else:
        lm_head = predictor_lm_head.predictor_head

    # Save Predictor heads
    torch.save(
        lm_head.state_dict(),
        os.path.join(training_args.output_dir, "predictor_lm_head.pt"),
    )


if __name__ == "__main__":
    train()
