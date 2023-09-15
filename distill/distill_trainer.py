import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from specInfer.generator import Generator
from specInfer.common import pad_to_2d
from enum import Enum
import random
from torch.utils.data import DataLoader


class SampleSource(Enum):
    Student = 1
    Teacher = 2
    Mix = 3


eval_cnt = 0


class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.teacher_model = teacher_model
        self.loss_model = "soft_only"
        self.generator = Generator(
            self.model, self.teacher_model, self.tokenizer, args.max_propose_num
        )
        self.train_step_cnt = 0

        # online related params
        self.mode = args.mode
        self.online_eval_interval = args.online_eval_interval
        self.online_update_interval = args.online_update_interval
        self.buffer = []
        self.alphas = []
        self.sample_steps = []

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
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

    def get_generated_ids(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        require_logits,
    ):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_scores=require_logits,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if require_logits:
                logits = torch.cat(
                    [score.unsqueeze(1) for score in outputs["scores"]], dim=1
                )
            else:
                logits = None
            return outputs["sequences"], logits

    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        if self.mode == "offline":
            return self.offline_training_step(model, inputs)
        elif self.mode == "online":
            return self.online_training_step(model, inputs)
        else:
            raise ValueError()

    def online_training_step(self, model, inputs):
        max_new_tokens = 128
        bsz = inputs["input_ids"].shape[0]
        assert (
            bsz == 1
        ), f"Does not support batch size > 1 in online setting, input batch size: {bsz}"
        assert (
            self.args.gradient_accumulation_steps == 1
        ), f"Does not support grad_acc > 1 in online setting, grad_acc: {self.args.gradient_accumulation_steps}"

        # remove any masking
        input_ids =  inputs["input_ids"][inputs["attention_mask"]].unsqueeze(0)
        # use speculative decoding to generate tokens
        output = self.generator.generate(input_ids,
                                         max_new_tokens)
        
        debug = False
        if debug:
            ref_generated = self.get_generated_ids(self.teacher_model, 
                                                self.tokenizer, 
                                                input_ids, 
                                                torch.ones_like(input_ids), 
                                                max_new_tokens, False)[0]
            print(ref_generated)
            print(self.tokenizer.batch_decode(ref_generated))
            print(output.output)
            print(output.alpha_sum)
            print(output.sample_steps)
            print("------")
        
        token_ids = torch.cat([input_ids, output.generated_ids], dim=-1)
        wrong_token_ids = [
            input_ids.shape[-1] + t for t in output.wrong_token_ids
        ]
        self.buffer.append((token_ids, wrong_token_ids))
        self.alphas.append(output.alpha_sum)
        self.sample_steps.append(output.sample_steps)

        if self.train_step_cnt % self.online_eval_interval == 0:
            window_size = 100
            avg_alpha = (
                sum(self.alphas[-window_size:])
                * 1.0
                / sum(self.sample_steps[-window_size:])
            )
            wandb.log({"alpha": avg_alpha})

        if len(self.buffer) >= self.online_update_interval:
            self.model.train()  # switch back to training mode

            input_ids = pad_to_2d([x[0] for x in self.buffer], 0)
            student_logits = self.get_logits(
                model, input_ids, torch.ones_like(input_ids)
            )
            # generate teacher logits as the label
            # TODO: we can avoid this forward by getting logits during speculative decoding
            with torch.no_grad():
                teacher_logits = self.get_logits(
                    self.teacher_model, input_ids, torch.ones_like(input_ids)
                )

            # only compute loss at wrong predictions
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            for i, data in enumerate(self.buffer):
                cur_wrong_token_ids = data[1]
                mask[i, cur_wrong_token_ids] = False

            loss = self.soft_cross_entropy(student_logits, teacher_logits, mask)
            loss.backward()
            self.buffer = []
            return loss.detach()
        else:
            return torch.tensor(-1)

    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']
        
        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        # Create custom DataLoader with shuffle set to False
        shuffle = False if self.mode == "online" else True
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))
        return DataLoader(self.train_dataset, shuffle=shuffle, batch_size=self.batch_size)
          
    def offline_training_step(self, model, inputs):
        max_new_tokens = 128
        temperature = 1
        sample_source = SampleSource.Teacher
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
        generated_ids, generated_logits = self.get_generated_ids(
            sample_model,
            self.tokenizer,
            inputs["input_ids"],
            inputs["attention_mask"],
            max_new_tokens,
            require_logits,
        )

        # prepare inputs for getting logits
        bsz, total_seq_len = generated_ids.shape
        gen_len = total_seq_len - inputs["input_ids"].shape[-1]
        attention_mask = torch.cat(
            [
                inputs["attention_mask"],
                torch.ones((bsz, gen_len), device="cuda", dtype=torch.long),
            ],
            dim=1,
        )

        # get student/teacher logits
        student_logits = self.get_logits(model, generated_ids, attention_mask)[
            :, -gen_len - 1 : -1, :
        ]
        with torch.no_grad():
            if generated_logits is not None:
                teacher_logits = generated_logits
            else:
                teacher_logits = self.get_logits(
                    self.teacher_model, generated_ids, attention_mask
                )[:, -gen_len - 1 : -1, :]

        # calculate loss with kl divergence
        output_mask = generated_ids[:, -gen_len:] == self.tokenizer.pad_token_id
        if kl_method == "teacher_student":
            loss = self.soft_cross_entropy(
                student_logits / temperature, teacher_logits / temperature, output_mask
            )
        elif kl_method == "student_teacher":
            loss = self.get_kl(
                teacher_logits / temperature, student_logits / temperature, output_mask
            )
        elif kl_method == "exact":
            vocab_size = teacher_logits.shape[-1]
            teacher_logits = teacher_logits.reshape(-1, vocab_size)
            student_logits = student_logits.reshape(-1, vocab_size)
            generated_ids = generated_ids[:, -gen_len:].reshape(-1, 1)
            with torch.no_grad():
                log_ratio = teacher_logits.log_softmax(-1).gather(
                    -1, generated_ids
                ) - student_logits.log_softmax(-1).gather(-1, generated_ids)
                log_ratio = log_ratio.reshape(bsz, gen_len).sum(dim=1)[:, None]
            cross_entropy = torch.nn.functional.cross_entropy(
                student_logits / temperature,
                generated_ids.squeeze(-1),
                ignore_index=self.tokenizer.pad_token_id,
                reduction="none",
            ).reshape(bsz, gen_len)
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
        if eval_cnt > 10 and eval_cnt % 5 != 0:
            return None, None, None

        output = self.generator.generate(inputs["input_ids"], 200)
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


class DistillTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.correct_cnt = 0
        self.propose_cnt = 0

        self.alpha = 0
        self.sample_steps = 0

    def on_evaluate(self, args, state, control, **kwargs):
        global eval_cnt
        print(f"[{eval_cnt}] {self.correct_cnt}/{self.propose_cnt}")

        if self.correct_cnt > 0:
            with open("out", "a") as f:
                f.write(f"[{eval_cnt}] {self.correct_cnt}/{self.propose_cnt}\n")
            wandb.log({"generated_token": self.correct_cnt * 1.0 / self.propose_cnt})
            wandb.log({"alpha": self.alpha * 1.0 / self.sample_steps})

        eval_cnt += 1
        self.correct_cnt = 0
        self.propose_cnt = 0

        self.alpha = 0
        self.sample_steps = 0
