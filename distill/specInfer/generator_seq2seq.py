import torch
from specInfer.common import (Seq2SeqOutputAndCache,
                              target_sample_from_distribution)
from specInfer.proposer_seq2seq import Seq2SeqSmallModelProposer, Seq2SeqSmallModelKVCacheProposer
from specInfer.verifier_seq2seq import Seq2SeqVerifier
from specInfer.common import sychronize_time, Seq2SeqInputAndCache
from specInfer.logger import SpecLogger
from dataclasses import dataclass
from typing import List

from .generator import GeneratorOutput, Generator

# logger = SpecLogger("output/generator.info")

class Seq2SeqGenerator(Generator):
    def __init__(self,
                 small_model,
                 large_model,
                 tokenizer,
                 max_propose_num,
                 use_cache=True) -> None:
        self.model = large_model
        self.tokenizer = tokenizer
        # metrics
        self.benchmark_time = False
        self.generation_time = []

        if use_cache:
            self.proposer = Seq2SeqSmallModelKVCacheProposer(
                small_model, tokenizer, self.benchmark_time)
        else:
            self.proposer = Seq2SeqSmallModelProposer(
                small_model, tokenizer, self.benchmark_time)
        self.verifier = Seq2SeqVerifier(large_model, tokenizer, self.benchmark_time)

        # parameters
        self.max_propose_num = max_propose_num
    
    @torch.inference_mode()
    def generate(self, input_ids, max_tokens, labels=None, temperature=0.01):
        def sample_method(logits):
            return torch.softmax(logits / temperature, dim=-1)

        generated_token_cnt = 0
        generated_tokens = None
        # generate 1 dummy decoder input ids
        proposer_input = Seq2SeqInputAndCache(
            input_ids, torch.Tensor([self.tokenizer.pad_token_id]).expand(input_ids.shape[0], 1).to(self.model.device).long(), labels, torch.ones_like(input_ids), None)
        verifier_input = Seq2SeqInputAndCache(
            input_ids, torch.Tensor([self.tokenizer.pad_token_id]).expand(input_ids.shape[0], 1).to(self.model.device).long(), labels, torch.ones_like(input_ids), None)

        correct_tokens = None
        propose_steps = 0
        alpha, sample_steps = 0, 0
        while True:
            start = sychronize_time()
            # propose n tokens, proposer always propose the token with highest probability
            proposer_output = self.proposer.propose(
                proposer_input,
                self.max_propose_num,
                sample_method)
            propose_steps += 1

            # prepare verifier input
            verifier_input = self.verifier.prepare_input(proposer_output,
                                                         verifier_input)
            # forward n tokens on the model in the a single run
            verifier_output = self.verifier.verify(
                verifier_input,
                proposer_output.generated_len,
                sample_method)

            # compare selected tokens
            # accept_token_ids, cur_alpha, cur_sample_steps = self.compare_tokens(proposer_output, verifier_output)
            accept_token_ids, cur_alpha, cur_sample_steps = self.sample_tokens(
                proposer_output, verifier_output)
            alpha += cur_alpha
            sample_steps += cur_sample_steps
            # logger.log("acc_tokens", accept_token_ids)
            if generated_tokens is None:
                generated_tokens = accept_token_ids
            else:
                generated_tokens = torch.cat(
                    [generated_tokens, accept_token_ids], dim=-1)
            generated_token_cnt += accept_token_ids.shape[1]
            if correct_tokens is None:
                correct_tokens = accept_token_ids[:, :-1]
            else:
                correct_tokens = torch.cat(
                    [correct_tokens, accept_token_ids[:, :-1]], dim=-1)

            # adjust the proposer/verifier input, discard unnecessary kv cache
            proposer_input = self.proposer.adjust_input(
                accept_token_ids, proposer_input, proposer_output)
            verifier_input = self.verifier.adjust_input(
                accept_token_ids, verifier_input, verifier_output)

            if self.benchmark_time:
                self.generation_time.append(sychronize_time() - start)

            if generated_token_cnt >= max_tokens or self.tokenizer.eos_token_id in accept_token_ids:
                break

        self.proposer.print_time()
        self.verifier.print_time()
        self.print_time()
        return GeneratorOutput(self.tokenizer.batch_decode(generated_tokens),
                               correct_tokens,
                               propose_steps,
                               sample_steps, alpha)
