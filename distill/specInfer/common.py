import torch
import time

from dataclasses import dataclass


@dataclass
class InputAndCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: torch.Tensor

@dataclass
class Seq2SeqInputAndCache:
    input_ids: torch.Tensor
    decoder_input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: torch.Tensor

@dataclass
class OutputAndCache:
    generated_len: int
    output_ids: torch.Tensor
    output_logits: torch.Tensor
    output_distribution: torch.Tensor
    past_key_values: torch.Tensor

@dataclass
class Seq2SeqOutputAndCache:
    generated_len: int
    output_ids: torch.Tensor
    output_logits: torch.Tensor
    output_distribution: torch.Tensor
    past_key_values: torch.Tensor


########################### Sampling ########################
def target_sample_from_distribution(target_distribution, draft_distribution):
    distribution = (target_distribution - draft_distribution)
    distribution = torch.max(distribution,
                             torch.zeros_like(distribution))
    if (distribution.sum(dim=-1, keepdim=True) == 0).any():
        distribution = torch.where(
            distribution == 0, distribution + 1e-10, distribution)
        print("[Warning] Distribution contains zero values")
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)

########################### Utility ########################


def slice_past_key_values(past_key_values, start_idx, slice_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
            (
                past_key_values[idx][0][:, :,
                                        start_idx:start_idx+slice_len, :],
                past_key_values[idx][1][:, :,
                                        start_idx:start_idx+slice_len, :],
            )
        )
    return tuple(new_past)


def slice_mqa_past_key_values(past_key_values, start_idx, slice_len):
    new_past = []
    for idx in range(len(past_key_values)):
        new_past.append(
            past_key_values[idx][:, start_idx:start_idx+slice_len, :]
        )
    return tuple(new_past)


def crop_past_key_values(past_key_values, max_len):
    return slice_past_key_values(past_key_values, 0, max_len)


def crop_mqa_past_key_values(past_key_values, max_len):
    return slice_mqa_past_key_values(past_key_values, 0, max_len)


def sychronize_time():
    torch.cuda.synchronize()
    return time.time()
