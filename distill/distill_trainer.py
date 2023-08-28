import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from specInfer.generator import Generator

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_model = "soft_only"
        self.eval_cnt = 0
        self.generator = Generator(self.model, self.teacher_model, self.tokenizer)
      
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = - targets_prob * student_likelihood
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy
    
    def training_step(self, model, inputs):
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
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        output, matches, propose_cnt = self.generator.generate(inputs["input_ids"], 200)
        find = False
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, DistillTrainerCallback):
                callback.correct_cnt += matches.shape[-1]
                callback.propose_cnt += propose_cnt
                find = True
        assert find

        return None, None, None

class DistillTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.eval_step = 0
        self.correct_cnt = 0
        self.propose_cnt = 0
        
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"[{self.eval_step}] {self.correct_cnt}/{self.propose_cnt}")
        with open("out", "a") as f:
            f.write(f"[{self.eval_step}] {self.correct_cnt}/{self.propose_cnt}\n")
        wandb.log({"eval_correctness": self.correct_cnt * 1.0/self.propose_cnt})
        self.eval_step += 1
        self.correct_cnt = 0
        self.propose_cnt = 0