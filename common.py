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
    past_key_values: torch.Tensor
    

########################### Utility ########################
def crop_past_key_values(past_key_values, max_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
        (
            past_key_values[idx][0][:, :, :max_len, :],
            past_key_values[idx][1][:, :, :max_len, :],
        )
        )
    return tuple(new_past)

def sychronize_time():
    torch.cuda.synchronize()
    return time.time()