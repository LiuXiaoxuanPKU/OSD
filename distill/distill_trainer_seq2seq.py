import torch
from transformers import Trainer, Seq2SeqTrainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from specInfer.generator import Generator
from specInfer.generator_seq2seq import Seq2SeqGenerator
from enum import Enum
import random

import typing

class SampleSource(Enum):
    Student = 1
    Teacher = 2
    Mix = 3

class Seq2SeqDistillTrainer(Seq2SeqTrainer):
    def __init__(self,
                 teacher_model,
                 propose_num,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_model = "soft_only"
        self.eval_cnt = 0
        self.generator = Seq2SeqGenerator(self.model,
                                   self.teacher_model,
                                   self.tokenizer,
                                   propose_num)

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = - targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_kl(self, predicts, targets, padding_mask):
        kl_loss = torch.nn.KLDivLoss(reduction="none")
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        output = kl_loss(predict_log_prob, targets_prob)
        expand_mask = padding_mask.unsqueeze(-1).expand_as(output)
        output.masked_fill_(expand_mask, 0)
        mean_output = output.sum() / (~padding_mask).sum()
        return mean_output
    
    def get_generated_ids(self, model, tokenizer,
                          input_ids, max_new_tokens, require_logits,
                          attention_mask: typing.Optional[torch.FloatTensor] = None):
        with torch.no_grad():
            if attention_mask != None:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    output_scores=require_logits,
                    return_dict_in_generate=True,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            elif attention_mask == None:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    output_scores=require_logits,
                    return_dict_in_generate=True,
                    pad_token_id=0,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                raise NotImplementedError()
            if require_logits:
                logits = torch.cat(
                    [score.unsqueeze(1) for score in outputs["scores"]], dim=1)
            else:
                logits = None
            return outputs["sequences"], logits

    def get_logits(self, model, input_ids, labels, attention_mask):
        return model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        ).logits
    
    def training_step(self, model, inputs):
        max_new_tokens = 128
        max_new_decoder_tokens = 32
        temperature = 1
        sample_source = SampleSource.Student
        kl_method = "teacher_student"

        # sample token ids
        if sample_source == SampleSource.Student:
            sample_model = model
        elif sample_source == SampleSource.Teacher:
            sample_model = self.teacher_model
        elif sample_source == SampleSource.Mix:
             l = random.random()
             sample_model = model if l < 0.5 else self.teacher_model
        else:
            raise ValueError()
        
        require_logits = True if sample_model == self.teacher_model else False

        generated_ids, generated_logits = self.get_generated_ids(sample_model,
                                                                 self.tokenizer,
                                                                 inputs['input_ids'],
                                                                 max_new_tokens,
                                                                 require_logits,
                                                                 attention_mask=inputs['attention_mask'])
        # prepare inputs for getting logits
        bsz, gen_seq_len = generated_ids.shape
        
        label_len = inputs['labels'].shape[1]
        # hard-coded embedding index fix
        input_labels = inputs['labels'].clone()

        bsz_label, input_label_len = input_labels.shape
        if gen_seq_len >= input_label_len:
            # pad label ids
            padding_len = gen_seq_len - input_label_len
            input_labels = torch.cat([input_labels,
                                    torch.ones((bsz, padding_len), device='cuda', dtype=torch.long)], dim=1)
            input_labels[:, -padding_len:] = self.tokenizer.pad_token_id
            
            attention_mask = inputs['attention_mask'][:, :gen_seq_len]
        else:
            # pad generated id
            padding_len = input_label_len - gen_seq_len
            generated_ids = torch.cat([generated_ids,
                                    torch.ones((bsz, padding_len), device='cuda', dtype=torch.long)], dim=1)
            generated_ids[:, -padding_len:] = self.tokenizer.pad_token_id

            attention_mask = inputs['attention_mask'][:, :input_label_len]

        # get student/teacher logits
        student_logits = self.get_logits(model, generated_ids, input_labels, attention_mask)[:, :, :]
        with torch.no_grad():
            if generated_logits is not None:
                teacher_logits = generated_logits
            else:
                teacher_logits = self.get_logits(self.teacher_model, generated_ids, input_labels, attention_mask)[
                    :, :, :]

        # calculate loss with kl divergence
        output_mask = generated_ids[:, :] == self.tokenizer.pad_token_id
        if kl_method == "teacher_student":
            loss = self.soft_cross_entropy(student_logits / temperature,
                                    teacher_logits / temperature,
                                    output_mask)
        elif kl_method == "student_teacher":
            loss = self.get_kl(teacher_logits / temperature,
                               student_logits / temperature,
                               output_mask)
        elif kl_method == "exact":
            vocab_size = teacher_logits.shape[-1]
            teacher_logits = teacher_logits.reshape(-1, vocab_size)
            student_logits = student_logits.reshape(-1, vocab_size)
            generated_ids = generated_ids[:, :].reshape(-1, 1)
            with torch.no_grad():
                log_ratio = (teacher_logits.log_softmax(-1).gather(-1, generated_ids) -
                            student_logits.log_softmax(-1).gather(-1, generated_ids))
                log_ratio = log_ratio.reshape(bsz, gen_seq_len).sum(dim=1)[:, None]
            cross_entropy = torch.nn.functional.cross_entropy(
                student_logits / temperature,
                generated_ids.squeeze(-1),
                ignore_index=self.tokenizer.pad_token_id,
                reduction='none').reshape(bsz, gen_seq_len)
            loss = cross_entropy * (log_ratio - 1)
            loss = (loss * (~output_mask)).sum() / (~output_mask).sum()
        else:
            raise NotImplementedError()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    @torch.inference_mode()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        output = self.generator.generate(inputs["input_ids"], input["labels"], 200)
        find = False
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, DistillTrainerCallback):
                callback.correct_cnt += output.correct_tokens.shape[-1]
                callback.propose_cnt += output.propose_steps
                callback.alpha += output.alpha_sum
                callback.sample_steps += output.sample_steps
                find = True
        assert find

        return None, None, None


class Seq2SeqDistillTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.eval_step = 0
        self.correct_cnt = 0
        self.propose_cnt = 0

        self.alpha = 0
        self.sample_steps = 0

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"[{self.eval_step}] {self.correct_cnt}/{self.propose_cnt}")
        with open("out", "a") as f:
            f.write(
                f"[{self.eval_step}] {self.correct_cnt}/{self.propose_cnt}\n")
        wandb.log({"generated_token": self.correct_cnt * 1.0 / self.propose_cnt})
        wandb.log({"alpha": self.alpha * 1.0 / self.sample_steps})

        self.eval_step += 1
        self.correct_cnt = 0
        self.propose_cnt = 0

        self.alpha = 0
        self.sample_steps = 0
