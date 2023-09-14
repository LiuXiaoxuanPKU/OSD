import time
import torch
from typing import Tuple
from specInfer.common import (InputAndCache,
                              OutputAndCache,
                              crop_past_key_values,
                              crop_mqa_past_key_values,
                              sychronize_time)
from transformers import LogitsProcessorList
import numpy as np


class Verifier:
    def __init__(self, model, tokenizer, benchmark_time=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_inputs = None
        self.processor = LogitsProcessorList()

        self.set_prompt_time = 0
        self.verify_times = []
        self.prepare_input_time = 0
        self.adjust_input_time = 0
        self.benchmark_time = benchmark_time

    def verify(self, input: InputAndCache,
               propose_len: int,
               sample_method) -> Tuple[InputAndCache, torch.Tensor]:
        if self.benchmark_time:
            start = sychronize_time()

        outputs = self.model(input_ids=input.input_ids,
                             attention_mask=input.attention_mask,
                             past_key_values=input.past_key_values,
                             use_cache=True)
        next_token_scores = self.processor(input.input_ids, outputs.logits)
        generated_len = propose_len + 1
        logits = next_token_scores[:, -generated_len:, :]
        # next_tokens = sample_fn(logits)

        if self.benchmark_time:
            self.verify_times.append(sychronize_time() - start)
        # output logits/distribution has shape [# of proposed tokens, vocab_size]
        # we squeeze the batch size dimension in the output because it is always 1
        return OutputAndCache(generated_len, None, logits.squeeze(0),
                              sample_method(logits.squeeze(0)), outputs.past_key_values)

    def prepare_input(self, proposer_output: OutputAndCache,
                      verifier_input: InputAndCache) -> InputAndCache:
        if self.benchmark_time:
            start = sychronize_time()

        if verifier_input.past_key_values is None:
            # concatenate proposed inputs with prompts
            input_ids = torch.cat(
                [verifier_input.input_ids, proposer_output.output_ids], dim=-1)
            # prompt phase, we don't have kv cache (past_key_values)
            past_key_values = None
            # concatenate prompt masks with proposed token masks
            attention_mask = torch.cat([verifier_input.attention_mask,
                                        torch.ones_like(proposer_output.output_ids,
                                                        dtype=torch.long, device="cuda")], dim=-1)
        else:
            input_ids = torch.cat([verifier_input.input_ids.unsqueeze(
                0), proposer_output.output_ids], dim=-1)
            past_key_values = verifier_input.past_key_values
            attention_mask = torch.cat([verifier_input.attention_mask,
                                        torch.ones_like(proposer_output.output_ids,
                                                        dtype=torch.long, device="cuda")], dim=-1)
        if self.benchmark_time:
            self.prepare_input_time += sychronize_time() - start
        return InputAndCache(input_ids, attention_mask, past_key_values)

    def adjust_input(self,
                     accept_token_ids: torch.Tensor,
                     verifier_input: InputAndCache,
                     verifier_output: OutputAndCache) -> InputAndCache:
        if self.benchmark_time:
            start = sychronize_time()

        n_matches = accept_token_ids.shape[1]
        verifier_input_ids = accept_token_ids[:, -1]

        if str(self.model.__class__.__name__) in ["GPTBigCodeForCausalLM"]:
            verifier_generated_len = verifier_output.past_key_values[0].shape[-2] - (
                verifier_output.generated_len - 1) + n_matches
            verifier_key_values = crop_mqa_past_key_values(
                verifier_output.past_key_values, verifier_generated_len - 1)
        else:
            verifier_generated_len = verifier_output.past_key_values[0][0].shape[2] - (
                verifier_output.generated_len - 1) + n_matches
            verifier_key_values = crop_past_key_values(
                verifier_output.past_key_values, verifier_generated_len - 1)

        verifier_attn_masks = verifier_input.attention_mask[:,
                                                            :verifier_generated_len]
        if verifier_attn_masks.shape[1] < verifier_generated_len:
            verifier_attn_masks = torch.cat([verifier_attn_masks,
                                             torch.ones(verifier_attn_masks.shape[0], 1, dtype=torch.long, device="cuda")], dim=-1)

        if self.benchmark_time:
            self.adjust_input_time += sychronize_time() - start
        return InputAndCache(verifier_input_ids, verifier_attn_masks, verifier_key_values)

    def print_time(self):
        if self.benchmark_time:
            print(f"[Verifier] prompt phase: {self.verify_times[0]}, "
                  f"decode phase: {np.median(self.verify_times[1:])}, ",
                  f"set prompt time: {self.set_prompt_time}, ",
                  f"adjust time: {self.adjust_input_time}, ",
                  f"prepare input time: {self.prepare_input_time}")
