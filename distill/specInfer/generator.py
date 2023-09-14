import torch
from specInfer.common import (OutputAndCache,
                              target_sample_from_distribution)
from specInfer.proposer import SmallModelProposer, SmallModelKVCacheProposer
from specInfer.verifier import Verifier
from specInfer.common import sychronize_time, InputAndCache
from specInfer.logger import SpecLogger
from dataclasses import dataclass
from typing import List

# logger = SpecLogger("output/generator.info")


@dataclass
class GeneratorOutput:
    output: List[str]
    generated_ids: torch.tensor
    correct_tokens: torch.tensor
    propose_steps: int
    sample_steps: int
    alpha_sum: float
    wrong_token_ids: List[int]


class Generator:
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
            self.proposer = SmallModelKVCacheProposer(
                small_model, tokenizer, self.benchmark_time)
        else:
            self.proposer = SmallModelProposer(
                small_model, tokenizer, self.benchmark_time)
        self.verifier = Verifier(large_model, tokenizer, self.benchmark_time)

        # parameters
        self.max_propose_num = max_propose_num

    def compare_tokens(self, proposed_output: OutputAndCache, verified_output: OutputAndCache) -> torch.Tensor:
        assert proposed_output.output_ids.shape == verified_output.output_ids[:, :-1].shape, \
            f"{proposed_output.output_ids.shape}, {verified_output.output_ids[:, :-1].shape}"

        # a = [[1, 2, 3]], b = [[1, 2, 4]]
        # ~(a == b): [[0, 0, 1]]
        # after cumsum: [[0, 0, 1]]
        # after < 1: [[1, 1, 0]]
        n_matches = ((~(proposed_output.output_ids ==
                     verified_output.output_ids[:, :-1])).cumsum(dim=-1) < 1).sum()
        return verified_output.output_ids[:, :n_matches + 1], -1, -1

    def sample_tokens(self,
                      proposed_output: OutputAndCache,
                      verified_output: OutputAndCache) -> torch.Tensor:
        # Accept-reject token loop
        accept_ids = []
        all_accepted = True
        sample_steps = 0
        alpha = 0
        for t in range(proposed_output.generated_len):
            sampled_ratios = (
                verified_output.output_distribution[t,
                                                    proposed_output.output_ids[0, t]]
                / proposed_output.output_distribution[t, proposed_output.output_ids[0, t]]
            )
            sampled_ratios = torch.min(sampled_ratios,
                                       torch.ones_like(sampled_ratios))
            rs = torch.rand_like(sampled_ratios)
            # logger.log("sample ratio", (rs, sampled_ratios))
            cur_alpha = min(verified_output.output_distribution[t, proposed_output.output_ids[0, t]],
                            proposed_output.output_distribution[t, proposed_output.output_ids[0, t]])

            assert cur_alpha >= 0 and cur_alpha <= 1
            alpha += cur_alpha
            sample_steps += 1

            if rs < sampled_ratios:
                accept_ids.append(proposed_output.output_ids[:, t])
            else:
                all_accepted = False
                next_token_id = target_sample_from_distribution(
                    verified_output.output_distribution[t, :],
                    proposed_output.output_distribution[t, :])
                accept_ids.append(next_token_id.unsqueeze(0))
                break

        # if all tokens were accepted, sample a last one
        if all_accepted:
            next_token_id = torch.multinomial(
                verified_output.output_distribution[-1, :], num_samples=1)

            assert next_token_id.dim() == 1
            accept_ids.append(next_token_id)

        accept_ids = torch.cat(accept_ids, dim=0)
        return accept_ids.unsqueeze(0), alpha, sample_steps

    @torch.inference_mode()
    def generate(self, input_ids, max_tokens, temperature=0.01):
        # make sure all models are in the inference mode
        self.model.eval()
        self.proposer.model.eval()
        
        def sample_method(logits):
            return torch.softmax(logits / temperature, dim=-1)

        generated_token_cnt = 0
        generated_tokens = None
        wrong_token_ids = []
        proposer_input = InputAndCache(
            input_ids, torch.ones_like(input_ids), None)
        verifier_input = InputAndCache(
            input_ids, torch.ones_like(input_ids), None)

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
            wrong_token_ids.append(generated_token_cnt - 1)

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
                               generated_tokens,
                               correct_tokens,
                               propose_steps,
                               sample_steps, 
                               alpha,
                               wrong_token_ids)

    def print_time(self):
        if self.benchmark_time:
            print(f"[Generator time]: {self.generation_time}")
            print(
                f"[Max allocated memory]: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
