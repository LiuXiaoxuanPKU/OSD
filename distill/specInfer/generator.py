import torch
from specInfer.common import OutputAndCache
from specInfer.proposer import SmallModelProposer
from specInfer.verifier import Verifier
from specInfer.common import sychronize_time, InputAndCache

import logging
logger = logging.getLogger('generator_logger') 
logger.setLevel(logging.WARNING) 
handler = logging.FileHandler('generator.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Generator:
    def __init__(self, small_model, large_model, tokenizer) -> None:
        self.model = large_model
        self.tokenizer = tokenizer
        self.proposer = SmallModelProposer(small_model, tokenizer)
        self.verifier = Verifier(large_model, tokenizer)
        
        # parameters
        self.max_propose_tokens = 5
        
        # metrics
        self.generation_time = []
        
    def compare_tokens(self, proposed_output: OutputAndCache, verified_output: OutputAndCache) -> torch.Tensor:
        assert proposed_output.output_ids.shape == verified_output.output_ids[:, :-1].shape, \
            f"{proposed_output.output_ids.shape}, {verified_output.output_ids[:, :-1].shape}"
        
        logger.debug(f"compare token: {proposed_output.output_ids}, {verified_output.output_ids}")
        # a = [[1, 2, 3]], b = [[1, 2, 4]]
        # ~(a == b): [[0, 0, 1]]
        # after cumsum: [[0, 0, 1]]
        # after < 1: [[1, 1, 0]]
        n_matches = ((~(proposed_output.output_ids == verified_output.output_ids[:, :-1])).cumsum(dim=-1) < 1).sum()
        return verified_output.output_ids[:, :n_matches + 1]

    @torch.inference_mode()
    def generate(self, input_ids, max_tokens):
        generated_token_cnt = 0
        generated_tokens = None
        proposer_input = InputAndCache(input_ids, torch.ones_like(input_ids), None)
        verifier_input = InputAndCache(input_ids, torch.ones_like(input_ids), None)
        
        correct_cnt = 0
        propose_steps = 0
        while True:
            start = sychronize_time()
            # propose n tokens
            proposer_output = self.proposer.propose(proposer_input, self.max_propose_tokens)
            propose_steps += 1
            
            # prepare verifier input
            verifier_input = self.verifier.prepare_input(proposer_output, 
                                                         verifier_input)
            
            # forward n tokens on the model in the a single run
            verifier_output = self.verifier.verify(verifier_input, proposer_output.generated_len)
            
            # compare selected tokens
            accept_token_ids = self.compare_tokens(proposer_output, verifier_output)
            logger.info(accept_token_ids.shape)
            if generated_tokens is None:
                generated_tokens = accept_token_ids
            else:
                generated_tokens = torch.cat([generated_tokens, accept_token_ids], dim=-1)
            generated_token_cnt += accept_token_ids.shape[1]
            correct_cnt += accept_token_ids.shape[1] - 1
            
            # adjust the proposer/verifier input, discard unnecessary kv cache
            proposer_input = self.proposer.adjust_input(accept_token_ids, proposer_input, proposer_output)
            verifier_input = self.verifier.adjust_input(accept_token_ids, verifier_input, verifier_output)
            
            self.generation_time.append(sychronize_time() - start)
            
            if generated_token_cnt >= max_tokens or self.tokenizer.eos_token_id in accept_token_ids:
                break
            
            logger.debug("================================")
        logger.debug(generated_tokens)
        logger.info(f"generated tokens: {generated_tokens.shape}")
        return self.tokenizer.batch_decode(generated_tokens), correct_cnt, propose_steps
    
    def __del__(self):
        # print(f"[Generator time: {self.generation_time}")
        print(f"[Max allocated memory]: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

