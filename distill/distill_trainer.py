import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from specInfer.generator import Generator
from enum import Enum


# class LossMethod(Enum):
#     TeacherKL = 1
#     StudentKL = 2
#     ReverseKL = 3


class DistillTrainer(Trainer):
    def __init__(self,
                 teacher_model,
                 propose_num,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_model = "soft_only"
        self.eval_cnt = 0
        self.generator = Generator(self.model,
                                   self.teacher_model,
                                   self.tokenizer,
                                   propose_num)

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = - targets_prob * student_likelihood
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def training_step_old(self, model, inputs):
        # Usual forward pass with the base model
        _, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        # Forward pass with the larger model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        padding_mask = inputs["labels"].eq(LabelSmoother.ignore_index)
        if self.loss_model == "soft_only":
            temperature = 1
            loss = self.soft_cross_entropy(student_outputs.logits / temperature,
                                           teacher_outputs.logits / temperature,
                                           padding_mask)
        else:
            raise NotImplementedError()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def get_generated_ids(self, model, input_ids, attention_mask, max_new_tokens):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
            logits = torch.cat(
                [score.unsqueeze(1) for score in outputs["scores"]], dim=1)
            return outputs["sequences"], logits

    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    
    def training_step(self, model, inputs):
        max_new_tokens = 128
        temperature = 1
        # method = LossMethod.TeacherKL

        sample_method = "teacher"
        kl_method = "teacher_student"

        # sample token ids
        sample_model = model if sample_method == "student" else self.teacher_model
        generated_ids, generated_logits = self.get_generated_ids(sample_model,
                                                                 inputs['input_ids'],
                                                                 inputs['attention_mask'],
                                                                 max_new_tokens)

        # prepare inputs for getting logits
        bsz, total_seq_len = generated_ids.shape
        gen_len = total_seq_len - inputs['input_ids'].shape[-1]
        output_mask = generated_ids != self.tokenizer.pad_token_id
        attention_mask = torch.cat([inputs['attention_mask'],
                                    torch.ones((bsz, gen_len), device='cuda', dtype=torch.long)], dim=1)

        # get student/teacher logits
        if sample_method == "teacher":
            teacher_logits = generated_logits
            student_logits = self.get_logits(model, generated_ids, attention_mask)[
                :, -gen_len-1:-1, :]
        else:
            with torch.no_grad():
                teacher_logits = self.get_logits(model, generated_ids, attention_mask)[
                    :, -gen_len-1:-1, :]
            student_logits = self.get_logits(
                self.teacher_model, generated_ids, attention_mask)[:, -gen_len-1:-1, :]

        # calculate loss with kl divergence
        output_mask = generated_ids[:, -gen_len:] != self.tokenizer.pad_token_id
        if kl_method == "teacher_student":
            self.soft_cross_entropy(student_logits / temperature,
                                    teacher_logits / temperature,
                                    output_mask)
        else:
            raise NotImplementedError()

        # if method == LossMethod.TeacherKL:
        #     with torch.no_grad():
        #         teacher_outputs = self.teacher_model.generate(
        #             input_ids=inputs['input_ids'],
        #             attention_mask=inputs['attention_mask'],
        #             max_new_tokens=max_new_tokens,
        #             output_scores=True,
        #             return_dict_in_generate=True,
        #         )
        #         teacher_logits = torch.cat(
        #             [score.unsqueeze(1) for score in teacher_outputs["scores"]], dim=1)

        #     bsz, total_seq_len = teacher_outputs["sequences"].shape
        #     gen_len = len(teacher_outputs["scores"])
        #     attention_mask = torch.cat([inputs['attention_mask'],
        #                                 torch.ones((bsz, gen_len), device='cuda', dtype=torch.long)], dim=1)
        #     student_outputs = model(
        #         input_ids=teacher_outputs.sequences,
        #         attention_mask=attention_mask)
        #     student_logits = student_outputs.logits[:, -gen_len-1:-1, :]

        #     use_logits = False
        #     if use_logits:
        #         mask = teacher_outputs["sequences"][:, -
        #                                             gen_len:] == self.tokenizer.pad_token_id
        #         loss = self.soft_cross_entropy(student_logits / temperature,
        #                                        teacher_logits / temperature,
        #                                        mask)
        #     else:
        #         vocab_size = student_logits.shape[-1]
        #         student_logits = student_logits.reshape(-1, vocab_size)
        #         generated_token_ids = teacher_outputs["sequences"][:, -
        #                                                            gen_len:].reshape(-1)
        #         loss = torch.nn.functional.cross_entropy(student_logits / temperature,
        #                                                  generated_token_ids,
        #                                                  ignore_index=self.tokenizer.pad_token_id)
        # elif method == LossMethod.StudentKL:
        #     with torch.no_grad():
        #         student_outputs = model.generate(
        #             input_ids=inputs['input_ids'],
        #             attention_mask=inputs['attention_mask'],
        #             max_new_tokens=max_new_tokens,
        #             return_dict_in_generate=True)

        #     bsz, total_seq_len = student_outputs["sequences"].shape
        #     gen_len = total_seq_len - inputs['input_ids'].shape[-1]
        #     generated_token_ids = student_outputs["sequences"][:, -gen_len:]
        #     output_mask = generated_token_ids != self.tokenizer.pad_token_id
        #     attention_mask = torch.cat([inputs['attention_mask'],
        #                                 torch.ones((bsz, gen_len), device='cuda', dtype=torch.long)], dim=1)
        #     student_logits = model(
        #         input_ids=student_outputs["sequences"],
        #         attention_mask=attention_mask,
        #     ).logits[:, -gen_len-1:-1, :]

        #     with torch.no_grad():
        #         teacher_outputs = self.teacher_model(
        #             input_ids=student_outputs["sequences"],
        #             attention_mask=attention_mask
        #         )
        #         teacher_logits = teacher_outputs.logits[:, -gen_len-1:-1, :]

        #     mask = student_outputs["sequences"][:, -
        #                                         gen_len:] == self.tokenizer.pad_token_id
        #     loss = self.soft_cross_entropy(student_logits / temperature,
        #                                    teacher_logits / temperature,
        #                                    mask)

        # elif method == LossMethod.ReverseKL:
        #     with torch.no_grad():
        #         student_outputs = model.generate(
        #             input_ids=inputs['input_ids'],
        #             attention_mask=inputs['attention_mask'],
        #             max_new_tokens=max_new_tokens,
        #             return_dict_in_generate=True)

        #     bsz, total_seq_len = student_outputs["sequences"].shape
        #     gen_len = total_seq_len - inputs['input_ids'].shape[-1]
        #     generated_token_ids = student_outputs["sequences"][:, -gen_len:]
        #     output_mask = generated_token_ids != self.tokenizer.pad_token_id
        #     attention_mask = torch.cat([inputs['attention_mask'],
        #                                 torch.ones((bsz, gen_len), device='cuda', dtype=torch.long)], dim=1)
        #     student_logits = model(
        #         input_ids=student_outputs["sequences"],
        #         attention_mask=attention_mask,
        #     ).logits[:, -gen_len-1:-1, :]

        #     with torch.no_grad():
        #         teacher_outputs = self.teacher_model(
        #             input_ids=student_outputs["sequences"],
        #             attention_mask=attention_mask
        #         )
        #         teacher_logits = teacher_outputs.logits[:, -gen_len-1:-1, :]
        #         vocab_size = student_logits.shape[-1]
        #         teacher_logits = teacher_logits.reshape(-1, vocab_size)
        #         student_logits = student_logits.reshape(-1, vocab_size)
        #         generated_token_ids = generated_token_ids.reshape(-1, 1)
        #         log_ratio = (teacher_logits.log_softmax(-1).gather(-1, generated_token_ids) -
        #                      student_logits.log_softmax(-1).gather(-1, generated_token_ids))
        #         log_ratio = log_ratio.reshape(bsz, gen_len).sum(dim=1)[:, None]

        #     cross_entropy = torch.nn.functional.cross_entropy(
        #         student_logits / temperature,
        #         generated_token_ids.squeeze(-1),
        #         ignore_index=self.tokenizer.pad_token_id,
        #         reduction='none').reshape(bsz, gen_len)
        #     print("cross_entropy", cross_entropy.shape)
        #     print("log_ratio", log_ratio.shape)
        #     loss = cross_entropy * (log_ratio - 1)
        #     loss = (loss * output_mask).sum() / output_mask.sum()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    @torch.inference_mode()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
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
