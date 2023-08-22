import time
import torch
from typing import Tuple
from common import InputAndCache, OutputAndCache, crop_past_key_values, sychronize_time
from transformers import LogitsProcessorList

import logging
logger = logging.getLogger('verifier_logger') 
logger.setLevel(logging.INFO) 

class Verifier:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_inputs = None
        self.processor = LogitsProcessorList()
        
        self.set_prompt_time = 0
        self.verify_time = 0
        self.prepare_input_time = 0
        self.adjust_input_time = 0
    
    def set_prompt(self, prompts) -> InputAndCache:
        start = sychronize_time()
        self.prompt_inputs = self.tokenizer(prompts, padding="longest", return_tensors="pt").to(device="cuda")
        logger.debug(f"prompt input length: {self.prompt_inputs.input_ids.shape}")
        self.set_prompt_time = sychronize_time() - start
        return InputAndCache(self.prompt_inputs.input_ids, self.prompt_inputs.attention_mask, None)
        
    def verify(self, input: InputAndCache, max_propose_tokens: int) -> Tuple[InputAndCache, torch.Tensor]:
        start = sychronize_time()
        
        outputs = self.model(input_ids=input.input_ids, 
                             attention_mask=input.attention_mask, 
                             past_key_values=input.past_key_values)
        next_token_scores = self.processor(input.input_ids, outputs.logits)
        generated_len = max_propose_tokens + 1
        next_tokens = torch.argmax(next_token_scores[:, -generated_len:, :], dim=-1)
        
        self.verify_time += sychronize_time() - start
        return OutputAndCache(generated_len, next_tokens, outputs.past_key_values)
    
    def prepare_input(self, proposer_output: OutputAndCache, 
                            verifier_input: InputAndCache) -> InputAndCache:
        logger.debug(proposer_output.output_ids.shape)
        start = sychronize_time()
        
        if verifier_input.past_key_values is None:
            # concatenate proposed inputs with prompts
            input_ids = torch.cat([self.prompt_inputs.input_ids, proposer_output.output_ids], dim=-1)
            # prompt phase, we don't have kv cache (past_key_values)
            past_key_values = None
            # concatenate prompt masks with proposed token masks
            attention_mask = torch.cat([self.prompt_inputs.attention_mask, 
                                        torch.ones_like(proposer_output.output_ids, 
                                                        dtype=torch.long, device="cuda")], dim=-1)
        else:
            input_ids = torch.cat([verifier_input.input_ids, proposer_output.output_ids], dim=-1)
            past_key_values = verifier_input.past_key_values
            attention_mask = torch.cat([verifier_input.attention_mask, 
                                        torch.ones_like(proposer_output.output_ids, 
                                                        dtype=torch.long, device="cuda")], dim=-1)
            
        self.prepare_input_time += sychronize_time() - start
        return InputAndCache(input_ids, attention_mask, past_key_values)
    
    def adjust_input(self, 
                     accept_token_ids: torch.Tensor, 
                     verifier_input: InputAndCache, 
                     verifier_output: OutputAndCache) -> InputAndCache:
        start = sychronize_time()
        
        n_matches = accept_token_ids.shape[1]
        verifier_input_ids = verifier_output.output_ids[:, n_matches-1:n_matches]
        verifier_generated_len = verifier_output.past_key_values[0][0].shape[2] - (verifier_output.generated_len - 1) + n_matches
        verifier_key_values = crop_past_key_values(verifier_output.past_key_values, verifier_generated_len - 1)
        
        verifier_attn_masks = verifier_input.attention_mask[:, :verifier_generated_len]
        if verifier_attn_masks.shape[1] < verifier_generated_len:
            verifier_attn_masks = torch.cat([verifier_attn_masks, 
                                        torch.ones(verifier_attn_masks.shape[0], 1, dtype=torch.long, device="cuda")], dim=-1)
            
        self.adjust_input_time += sychronize_time() - start
        return InputAndCache(verifier_input_ids, verifier_attn_masks, verifier_key_values)
    
    def __del__(self):
        print(f"[Verifier] verify time: {self.verify_time},",
              f"set prompt time: {self.set_prompt_time}",
              f"adjust time: {self.adjust_input_time}, prepare input time: {self.prepare_input_time}")
        
        
class OptimizeVerifier(Verifier):
    def prepare_input(self, proposer_output: OutputAndCache, 
                            verifier_input: InputAndCache) -> InputAndCache:
        pass
    
    def adjust_input(self, 
                     accept_token_ids: torch.Tensor, 
                     verifier_input: InputAndCache, 
                     verifier_output: OutputAndCache) -> InputAndCache:
        pass
    
    def set_prompt(self, prompts) -> InputAndCache:
        start = sychronize_time()
        self.prompt_inputs = self.tokenizer(prompts, padding="longest", return_tensors="pt").to(device="cuda")
        print(f"prompt input length: {self.prompt_inputs.input_ids.shape}")
        outputs = self.model(input_ids=self.prompt_inputs.input_ids, 
                             attention_mask=self.prompt_inputs.attention_mask, 
                             past_key_values=None)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        bsz = self.prompt_inputs.attention_mask.shape[0]
        attention_mask = torch.cat([self.prompt_inputs.attention_mask, 
                                    torch.ones(bsz, 1, dtype=torch.long, device="cuda")], dim=-1)
        self.set_prompt_time = sychronize_time() - start
        return InputAndCache(next_token, attention_mask, outputs.past_key_values)