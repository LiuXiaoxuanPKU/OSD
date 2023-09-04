import torch
import time

from dataclasses import dataclass

@dataclass
class InputAndCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: torch.Tensor

@dataclass
class OutputAndCache:
    generated_len: int
    output_ids: torch.Tensor
    output_logits: torch.Tensor
    past_key_values: torch.Tensor
    

########################### Sampling ########################
def get_temperature_distribution(logits, temperature):
    return torch.softmax(logits / temperature, dim=-1)

def sample_fn(logits, temperature):
    distribution = get_temperature_distribution(logits, temperature)
    if distribution.dim() > 2:
        distribution = distribution.squeeze(0)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)

def target_sample_from_distribution(target_distribution, draft_distribution):
    distribution = (target_distribution - draft_distribution)
    distribution = torch.max(distribution,
                             torch.zeros_like(distribution))
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)

########################### Utility ########################
def slice_past_key_values(past_key_values, start_idx, slice_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
        (
            past_key_values[idx][0][:, :, start_idx:start_idx+slice_len, :],
            past_key_values[idx][1][:, :, start_idx:start_idx+slice_len, :],
        )
        )
    return tuple(new_past)

def crop_past_key_values(past_key_values, max_len):
    return slice_past_key_values(past_key_values, 0, max_len)
def sychronize_time():
    torch.cuda.synchronize()
    return time.time()