import torch
from typing import List
from specInfer.common import (Seq2SeqInputAndCache,
                              Seq2SeqOutputAndCache,
                              crop_past_key_values_seq2seq,
                              crop_mqa_past_key_values,
                              sychronize_time)
import numpy as np

from .proposer import Proposer

class Seq2SeqProposer(Proposer):
    def __init__(self, benchmark_time=False) -> None:
        super().__init__(benchmark_time)

    # sample_method:
    #   input: logits
    #   output: probability distribution
    def propose(self,
                input: Seq2SeqInputAndCache,
                n: int,
                sample_method) -> Seq2SeqOutputAndCache:
        if self.benchmark_time:
            start = sychronize_time()

        ret = self.propose_impl(input, n, sample_method)

        if self.benchmark_time:
            self.propose_times.append(sychronize_time() - start)

        return ret

class Seq2SeqSmallModelProposer(Seq2SeqProposer):
    def __init__(self, model, tokenizer, benchmark_time) -> None:
        super().__init__(benchmark_time)
        self.model = model
        self.tokenizer = tokenizer

    def propose_impl(self,
                     input: Seq2SeqInputAndCache,
                     n: int,
                     sample_method) -> Seq2SeqOutputAndCache:
        if input.input_ids.shape[0] > 1:
            raise NotImplementedError(
                "Not implement for batch_size > 1 in evaluation")

        propose_tokens = []
        propose_distributions = []
        input_ids = input.input_ids
        decoder_input_ids = input.decoder_input_ids
        attention_mask = input.attention_mask
        labels = input.labels
        generated_len = n
        for i in range(n):
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            # next_token_logits has shape [1, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]
            distribution = sample_method(next_token_logits)
            next_token_id = torch.multinomial(distribution, num_samples=1)

            propose_distributions.append(distribution)
            propose_tokens.append(next_token_id)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                generated_len = i + 1
                # print(f"[Info] Stop at {generated_len} because of eos")
                break

            decoder_input_ids = torch.cat(
                    [decoder_input_ids, next_token_id.reshape(1, 1)], dim=-1)
        propose_tokens = torch.cat(propose_tokens, dim=-1)
        propose_distributions = torch.cat(propose_distributions, dim=0)
        return Seq2SeqOutputAndCache(generated_len, propose_tokens,
                              None, propose_distributions,
                              None)

    def adjust_input_impl(self, accept_token_ids: torch.Tensor,
                          proposer_input: Seq2SeqInputAndCache,
                          proposer_output: Seq2SeqOutputAndCache) -> Seq2SeqInputAndCache:
        new_decoder_input_ids = torch.cat(
            [proposer_input.decoder_input_ids, accept_token_ids], dim=-1)
        return Seq2SeqInputAndCache(input_ids, new_decoder_input_ids, labels, torch.ones_like(input_ids), None)

class Seq2SeqSmallModelKVCacheProposer(Seq2SeqProposer):
    def __init__(self, model, tokenizer, benchmark_time) -> None:
        super().__init__(benchmark_time)
        self.model = model
        self.tokenizer = tokenizer

    def propose_impl(self,
                     input: Seq2SeqInputAndCache,
                     n: int,
                     sample_method) -> Seq2SeqOutputAndCache:
        if input.input_ids.shape[0] > 1:
            raise NotImplementedError(
                "Not implement for batch_size > 1 in evaluation")

        propose_tokens = []
        propose_logits = []
        propose_distributions = []
        input_ids = input.input_ids
        decoder_input_ids = input.decoder_input_ids
        labels = input.labels
        attention_mask = input.attention_mask
        past_key_values = input.past_key_values
        generated_len = n
        for i in range(n):
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, 
                                attention_mask=attention_mask, 
                                past_key_values=past_key_values,
                                use_cache=True)
            past_key_values = outputs.past_key_values
            # next_token_logits has shape [1, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]
            distribution = sample_method(next_token_logits)
            next_token_id = torch.multinomial(distribution, num_samples=1)

            propose_logits.append(next_token_logits)
            propose_distributions.append(distribution)
            propose_tokens.append(next_token_id)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                generated_len = i + 1
                # print(f"[Info] Stop at {generated_len} because of eos")
                break 
            decoder_input_ids = next_token_id.reshape(1, 1)
        propose_tokens = torch.cat(propose_tokens, dim=-1)
        propose_logits = torch.cat(propose_logits, dim=0)
        propose_distributions = torch.cat(propose_distributions, dim=0)
        return Seq2SeqOutputAndCache(generated_len, propose_tokens,
                              propose_logits, propose_distributions,
                              past_key_values)

    def adjust_input_impl(self, accept_token_ids: torch.Tensor,
                          proposer_input: Seq2SeqInputAndCache,
                          proposer_output: Seq2SeqOutputAndCache) -> Seq2SeqInputAndCache:
        proposer_decoder_input_ids = accept_token_ids.tile(
            proposer_input.decoder_input_ids.shape[0], 1)

        # mqa
        if str(self.model.__class__.__name__) in ["GPTBigCodeForCausalLM"]:
            total_generated_len = proposer_output.past_key_values[0].shape[-2] + 1
            proposer_key_values = crop_mqa_past_key_values(proposer_output.past_key_values,
                                                           max_len=total_generated_len - proposer_output.generated_len)
        else:  # mha
            # substract one dummy pad token
            total_generated_len = proposer_output.past_key_values[0][0].shape[2] + 1
            proposer_key_values = crop_past_key_values_seq2seq(proposer_output.past_key_values,
                                                       max_len=total_generated_len - proposer_output.generated_len)

        return Seq2SeqInputAndCache(proposer_input.input_ids, proposer_decoder_input_ids, proposer_input.labels, proposer_input.attention_mask, proposer_key_values)