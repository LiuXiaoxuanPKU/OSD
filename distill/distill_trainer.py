import torch
from transformers import Trainer

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_model = "soft_only"
        
        self.eval_step = 0
        self.correct_cnt = 0
        self.total_cnt = 0
      
    @staticmethod  
    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()
    
    def training_step(self, model, inputs):
        # Usual forward pass with the base model
        _, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Forward pass with the larger model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
            if self.loss_model == "soft_only":
                temperature = 1
                loss = DistillTrainer.soft_cross_entropy(student_outputs.logits / temperature,
                                                        teacher_outputs.logits / temperature)
            else:
                raise NotImplementedError()
        
        del student_outputs, teacher_outputs
        return loss
    
    def prediction_step_example(self, model, inputs, prediction_loss_only, ignore_keys=None):
        n = 20
        with torch.no_grad():
            def get_correct_token(input_ids):
                if input_ids.shape[0] > 1:
                    raise NotImplementedError("Not implement for batch_size > 1 in evaluation")
                ######### propose #########
                propose_tokens = []
                i = 0
                student_input_ids = input_ids
                while True:
                    outputs = model(student_input_ids)
                    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                    propose_tokens.append(next_tokens[0].item())
                    i += 1     
                    if next_tokens[0] == self.tokenizer.eos_token_id or i == n:
                        break
                    student_input_ids = torch.cat([student_input_ids, next_tokens.reshape(1, 1)], dim=-1)
                propose_tokens = torch.tensor(propose_tokens, device=input_ids.device).reshape(1, -1)
                
                ######### verify #########
                # prepare input
                input_ids = torch.cat([input_ids, propose_tokens], dim=-1)
                # batch inference
                outputs = self.teacher_model(input_ids=input_ids)
                verifier_tokens = torch.argmax(outputs.logits[:, -(i+1):, :], dim=-1)
            
                ######### get correct tokens #########
                # a = [[1, 2, 3]], b = [[1, 2, 4]]
                # ~(a == b): [[0, 0, 1]]
                # after cumsum: [[0, 0, 1]]
                # after < 1: [[1, 1, 0]]
                n_matches = ((~(propose_tokens == verifier_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
                return n_matches
    
            correct_cnt = get_correct_token(inputs['input_ids'])
            # avg_correct_cnt = correct_cnt / inputs['input_ids'].shape[0]
            # print(f"average correct count: {avg_correct_cnt}/{n}")
            self.correct_cnt += correct_cnt
            self.total_cnt += n
            self.eval_step += 1
            if self.eval_step % 100 == 0:
                print(f"[{self.eval_step}] {self.correct_cnt}/{self.total_cnt}")
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)