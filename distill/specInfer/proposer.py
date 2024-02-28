import torch
from typing import List
from specInfer.common import (InputAndCache,
                              OutputAndCache,
                              ConfInfo,
                              crop_past_key_values,
                              crop_mqa_past_key_values,
                              crop_past_key_values_seq2seq,
                              sychronize_time)
import numpy as np
import torch.nn.functional as F

class Proposer:
    def __init__(self, benchmark_time=False) -> None:
        self.propose_times = []
        self.adjust_time = 0

        self.benchmark_time = benchmark_time

    # sample_method:
    #   input: logits
    #   output: probability distribution
    def propose(self,
                input: InputAndCache,
                n: int,
                sample_method) -> OutputAndCache:
        if self.benchmark_time:
            start = sychronize_time()

        ret = self.propose_impl(input, n, sample_method)

        if self.benchmark_time:
            self.propose_times.append(sychronize_time() - start)

        return ret

    def adjust_input(self, accept_token_ids: torch.Tensor,
                     proposer_input: InputAndCache,
                     proposer_output: OutputAndCache) -> InputAndCache:
        if self.benchmark_time:
            start = sychronize_time()

        ret = self.adjust_input_impl(
            accept_token_ids, proposer_input, proposer_output)

        if self.benchmark_time:
            self.adjust_time += sychronize_time() - start

        return ret

    def print_time(self):
        if self.benchmark_time:
            print(f"[Proposer] prompt phase: {self.propose_times[0]},",
                  f" decode phase: {np.median(self.propose_times[1:])}, ",
                  f" adjust time: {self.adjust_time}")


class RandomProposer(Proposer):
    def __init__(self, benchmark_time) -> None:
        super().__init__(benchmark_time)

    def set_prompt(self, prompts: List[str], past_key_values: torch.Tensor) -> InputAndCache:
        pass

    def propose_impl(self, input: InputAndCache, n: int) -> OutputAndCache:
        return OutputAndCache(1, torch.randint(0, 32000, (1, 16), device='cuda'), None)

    def adjust_input_impl(self, accept_token_ids: torch.Tensor,
                          proposer_input: InputAndCache,
                          proposer_output: OutputAndCache) -> InputAndCache:
        return InputAndCache(None, None, None)


class SmallModelProposer(Proposer):
    def __init__(self, model, tokenizer, is_encoder_decoder, benchmark_time) -> None:
        super().__init__(benchmark_time)
        self.model = model
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

    def propose_impl(self,
                     input: InputAndCache,
                     n: int,
                     sample_method) -> OutputAndCache:
        if input.input_ids.shape[0] > 1:
            raise NotImplementedError(
                "Not implement for batch_size > 1 in evaluation")

        propose_tokens = []
        propose_distributions = []
        input_ids = input.input_ids
        decoder_input_ids = input.decoder_input_ids
        generated_len = n
        for i in range(n):
            if self.is_encoder_decoder:
                outputs = self.model(input_ids=input_ids,
                                     decoder_input_ids=decoder_input_ids,
                                     attention_mask=input.attention_mask)
            else:
                outputs = self.model(input_ids)
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

            if self.is_encoder_decoder:
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, next_token_id.reshape(1, 1)], dim=-1)
            else:
                input_ids = torch.cat(
                    [input_ids, next_token_id.reshape(1, 1)], dim=-1)
        propose_tokens = torch.cat(propose_tokens, dim=-1)
        propose_distributions = torch.cat(propose_distributions, dim=0)
        return OutputAndCache(generated_len, propose_tokens,
                              None, propose_distributions,
                              None)

    def adjust_input_impl(self, accept_token_ids: torch.Tensor,
                          proposer_input: InputAndCache,
                          proposer_output: OutputAndCache) -> InputAndCache:
        if self.is_encoder_decoder:
            new_decoder_input_ids = torch.cat(
                [proposer_input.decoder_input_ids, accept_token_ids], dim=-1)
            return InputAndCache(proposer_input.input_ids,
                                 torch.ones_like(proposer_input.input_ids),
                                 None,
                                 proposer_input.labels,
                                 new_decoder_input_ids)
        else:
            new_input_ids = torch.cat(
                [proposer_input.input_ids, accept_token_ids], dim=-1)
            return InputAndCache(new_input_ids, torch.ones_like(new_input_ids), None)


class SmallModelKVCacheProposer(Proposer):
    def __init__(self, model, tokenizer, is_encoder_decoder, benchmark_time) -> None:
        super().__init__(benchmark_time)
        self.model = model
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

    def propose_impl(self,
                     input: InputAndCache,
                     n: int,
                     sample_method) -> OutputAndCache:
        if input.input_ids.shape[0] > 1:
            raise NotImplementedError(
                "Not implement for batch_size > 1 in evaluation")

        propose_tokens = []
        propose_logits = []
        propose_distributions = []
        propose_conf_infos = []
        input_ids = input.input_ids
        decoder_input_ids = input.decoder_input_ids
        past_key_values = input.past_key_values
        generated_len = n
        for i in range(n):
            if self.is_encoder_decoder:
                outputs = self.model(input_ids=input_ids,
                                     decoder_input_ids=decoder_input_ids,
                                     attention_mask=input.attention_mask,
                                     past_key_values=past_key_values,
                                     use_cache=True)
            else:
                outputs = self.model(input_ids,
                                     past_key_values=past_key_values,
                                     output_hidden_states=True,
                                     use_cache=True)
            past_key_values = outputs.past_key_values
            # next_token_logits has shape [1, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]
            distribution = sample_method(next_token_logits)
            next_token_id = torch.multinomial(distribution, num_samples=1)

            propose_logits.append(next_token_logits)
            propose_distributions.append(distribution)
            propose_tokens.append(next_token_id)

            ################################# Start calculating confidence related features #####################
            hidden_states = outputs.hidden_states
            last_token_states = []
            for h in hidden_states:
                last_token_states.append(h[:, -1, :])
            last_token_states = torch.cat(last_token_states, dim=0)
            all_layer_logits = self.model.lm_head(last_token_states)
            layer_probs = F.softmax(all_layer_logits, dim=-1)
            token_layer_probs = layer_probs[:, next_token_id].reshape(1, -1)
            # FIXME: The following only works for argmax sampling
            layer_token_ids = all_layer_logits.argmax(dim=-1)

            token_prob = token_layer_probs[0, -1].item()
            token_entropy = -torch.sum(layer_probs[-1, :] * torch.log(layer_probs[-1, :]), dim=-1).item()
            conf_info = ConfInfo(next_token_id, token_layer_probs,
                                 layer_token_ids, token_prob, token_entropy,
                                 None, None, None, None)
            propose_conf_infos.append(conf_info)
            ######################################################################################################

            if next_token_id.item() == self.tokenizer.eos_token_id:
                generated_len = i + 1
                # print(f"[Info] Stop at {generated_len} because of eos")
                break
            if self.is_encoder_decoder:
                decoder_input_ids = next_token_id.reshape(1, 1)
            else:
                input_ids = next_token_id
        propose_tokens = torch.cat(propose_tokens, dim=-1)
        propose_logits = torch.cat(propose_logits, dim=0)
        propose_distributions = torch.cat(propose_distributions, dim=0)
        return OutputAndCache(generated_len, propose_tokens,
                              propose_logits, propose_distributions,
                              past_key_values, propose_conf_infos)

    def adjust_input_impl(self, accept_token_ids: torch.Tensor,
                          proposer_input: InputAndCache,
                          proposer_output: OutputAndCache) -> InputAndCache:
        if self.is_encoder_decoder:
            proposer_decoder_input_ids = accept_token_ids.tile(
                proposer_input.decoder_input_ids.shape[0], 1)
            crop_func = crop_past_key_values_seq2seq
            crop_mqa_func = crop_mqa_past_key_values  # TODO: is it correct?
        else:
            proposer_input_ids = accept_token_ids.tile(
                proposer_input.input_ids.shape[0], 1)
            proposer_attn_masks = torch.cat([proposer_input.attention_mask,
                                             torch.ones_like(proposer_input_ids, dtype=torch.long)], dim=-1)
            crop_func = crop_past_key_values
            crop_mqa_func = crop_mqa_past_key_values  # TODO: is it correct?

        # mqa
        if str(self.model.__class__.__name__) in ["GPTBigCodeForCausalLM"]:
            total_generated_len = proposer_output.past_key_values[0].shape[-2] + 1
            proposer_key_values = crop_mqa_func(proposer_output.past_key_values,
                                                max_len=total_generated_len - proposer_output.generated_len)
        else:  # mha
            total_generated_len = proposer_output.past_key_values[0][0].shape[2] + 1
            proposer_key_values = crop_func(proposer_output.past_key_values,
                                            max_len=total_generated_len - proposer_output.generated_len)

        if self.is_encoder_decoder:
            return InputAndCache(proposer_input.input_ids,
                                 proposer_input.attention_mask,
                                 proposer_key_values,
                                 proposer_input.labels,
                                 proposer_decoder_input_ids)
        else:
            return InputAndCache(proposer_input_ids,
                                 proposer_attn_masks,
                                 proposer_key_values)
