import torch

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
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Forward pass with the larger model
        with torch.no_grad():
            large_outputs = self.large_model(**inputs)
        
        if self.loss_model == "soft_only":
            temperature = 1
            loss = DistillTrainer.soft_cross_entropy(student_logits / temperature,
                                                      teacher_logits / temperature)
        else:
            raise NotImplementedError()
        
        return loss