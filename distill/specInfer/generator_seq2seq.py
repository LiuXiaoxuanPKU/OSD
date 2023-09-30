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
@dataclass
class Seq2SeqGeneratorOutput:
    output: List[str]
    generated_ids: torch.tensor
    student_generated_ids: torch.tensor
    correct_tokens: torch.tensor
    propose_steps: int
    sample_steps: int
    alpha_sum: float
    wrong_token_ids: List[int]

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
    def generate(self, input_ids, attention_mask, max_tokens, labels=None, temperature=0.01):
        def sample_method(logits):
            return torch.softmax(logits / temperature, dim=-1)
        
        # make sure all models are in the inference mode
        self.model.eval()
        self.proposer.model.eval()

        generated_token_cnt = 0
        student_generated_token_cnt = 0
        generated_tokens = None
        student_generated_tokens = None
        wrong_token_ids = []
        # generate 1 dummy decoder input ids
        proposer_input = Seq2SeqInputAndCache(
            input_ids, torch.Tensor([self.model.config.decoder_start_token_id]).expand(input_ids.shape[0], 1).to(self.model.device).long(), labels, attention_mask, None)
        verifier_input = Seq2SeqInputAndCache(
            input_ids, torch.Tensor([self.model.config.decoder_start_token_id]).expand(input_ids.shape[0], 1).to(self.model.device).long(), labels, attention_mask, None)

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

            student_token_ids = proposer_output.output_ids
            if student_generated_tokens is None:
                student_generated_tokens = proposer_output.output_ids.reshape(1, -1)
            else:
                student_generated_tokens = torch.cat(
                    [student_generated_tokens, proposer_output.output_ids.reshape(1, -1)], dim=-1)
            student_generated_token_cnt += student_token_ids.shape[1]

            if student_token_ids.shape[1] < accept_token_ids.shape[1]:
                # append free token from teacher
                diff_count = accept_token_ids.shape[1] - student_token_ids.shape[1]
                student_generated_tokens = torch.cat([student_generated_tokens, accept_token_ids[:, -diff_count].reshape(1, -1)], dim=-1)
            if student_token_ids.shape[1] > accept_token_ids.shape[1]:
                # append free token from teacher
                diff_count = student_token_ids.shape[1] - accept_token_ids.shape[1]
                student_generated_tokens = student_generated_tokens[:, :-diff_count]

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
        return Seq2SeqGeneratorOutput(self.tokenizer.batch_decode(generated_tokens),
                                generated_tokens,
                                student_generated_tokens,
                                correct_tokens,
                                propose_steps,
                                sample_steps, 
                                alpha.item(),
                                wrong_token_ids)
