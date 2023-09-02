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