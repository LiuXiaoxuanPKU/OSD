import torch
from transformers import Trainer

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_model = "soft_only"
      
    @staticmethod  
    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()
    
    def training_step(self, model, inputs):
        # Usual forward pass with the base model
        loss, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Forward pass with the larger model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        if self.loss_model == "soft_only":
            temperature = 1
            loss = DistillTrainer.soft_cross_entropy(student_outputs.logits / temperature,
                                                      teacher_outputs.logits / temperature)
        else:
            raise NotImplementedError()
        
        return loss
    
    @torch.inference_mode()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        n = 20
        def get_correct_token(input):
            ######### propose #########
            propose_tokens = []
            i = 0
            while True:
                outputs = model(inputs)
                next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                propose_tokens.append(next_tokens[0].item())
                i += 1     
                if next_tokens[0] == self.tokenizer.eos_token_id:
                    break
            propose_tokens = torch.tensor(propose_tokens)
                
            ######### verify #########
            # prepare input
            input_ids = torch.cat([input.input_ids, propose_tokens], dim=-1)
            # batch inference
            outputs = self.model(input_ids=input_ids)
            verifier_tokens = torch.argmax(outputs.logits[:, -i:, :], dim=-1)
        
            ######### get correct tokens #########
            assert propose_tokens.shape == verifier_tokens.shape, \
            f"{propose_tokens.shape}, {verifier_tokens.shape}"
        
            # a = [[1, 2, 3]], b = [[1, 2, 4]]
            # ~(a == b): [[0, 0, 1]]
            # after cumsum: [[0, 0, 1]]
            # after < 1: [[1, 1, 0]]
            n_matches = ((~(proposed_output.output_ids == verified_output.output_ids[:, :-1])).cumsum(dim=-1) < 1).sum()
            n_matches = proposed_output.output_ids.shape[-1]
            return n_matches
   
        correct_cnt = 0
        for input in inputs:
            correct_cnt += get_correct_token()
        avg_correct_cnt = correct_cnt / inputs.shape[0]
        print(f"average correct count: {avg_correct_cnt}/{n}")
    