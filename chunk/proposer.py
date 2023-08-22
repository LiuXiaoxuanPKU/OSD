import torch
import time

from transformers import TopPLogitsWarper, LogitsProcessorList
from typing import Tuple, List
from common import InputAndCache, OutputAndCache, crop_past_key_values, sychronize_time

import logging
logger = logging.getLogger('proposer_logger') 
logger.setLevel(logging.INFO)


class Proposer:
    def __init__(self) -> None:
        self.propose_time = 0
        self.adjust_time = 0
    
    def set_prompt(self, prompts: List[str]) -> InputAndCache:
        pass
    
    def propose(self, input: InputAndCache, n: int) -> OutputAndCache:
        start = sychronize_time()
            
        ret = self.propose_impl(input, n)
        
        self.propose_time += sychronize_time() - start
        
        return ret
    
    def adjust_input(self, accept_token_ids: torch.Tensor, 
                     proposer_input: InputAndCache, 
                     proposer_output: OutputAndCache) -> InputAndCache:
        start = sychronize_time()
        
        ret = self.adjust_input_impl(accept_token_ids, proposer_input, proposer_output)
        
        self.adjust_time += sychronize_time() - start
        
        return ret
    
    def __del__(self):
        print(f"[Proposer] propose time: {self.propose_time}, adjust time: {self.adjust_time}")


class RandomProposer(Proposer):
    def __init__(self) -> None:
        super().__init__()
    
    def set_prompt(self, prompts: List[str], past_key_values: torch.Tensor) -> InputAndCache:
        pass
    
    def propose_impl(self, input: InputAndCache, n: int) -> OutputAndCache:
        return OutputAndCache(1, torch.randint(0, 32000, (1, 16), device='cuda'), None)
    
    def adjust_input_impl(self, accept_token_ids: torch.Tensor, 
                     proposer_input: InputAndCache, 
                     proposer_output: OutputAndCache) -> InputAndCache:
        return InputAndCache(None, None, None)
 
    
class NBCEProposer(Proposer):
    def __init__(self, model, tokenizer, chunker) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.processors = LogitsProcessorList()
        self.processors.append(TopPLogitsWarper(0.95))
    
    def set_prompt(self, prompts: List[str], past_key_values: torch.Tensor) -> InputAndCache:
        if len(prompts) > 1:
            raise NotImplementedError()
        prompt = prompts[0]
    
        chunked_prompts = self.chunker.chunk(prompt)
        inputs = self.tokenizer([''] + chunked_prompts, padding='longest', return_tensors='pt').to('cuda')
        
        return InputAndCache(inputs.input_ids, inputs.attention_mask, None)
    
    def propose_impl(self, input: InputAndCache, n: int) -> OutputAndCache:
        input_ids = input.input_ids
        past_key_values = input.past_key_values
        attention_mask = input.attention_mask
        batch_size = input_ids.shape[0]
        selected_tokens = torch.zeros(n, dtype=torch.long, device=input_ids.device)
        logger.debug(f"proposer input shape: {input_ids.shape}")
        logger.debug(f"proposer attention_mask shape {attention_mask.shape}")
        if past_key_values is not None:
            logger.debug(f"proposer past_key_values shape: {past_key_values[0][0].shape}")
        generated_len = n
        for i in range(n):
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            
            # ========== NBCE ==========
            beta, eta = 0.25, 0.1
            logits = outputs.logits[:, -1]
            logits = logits - logits.logsumexp(dim=-1, keepdims=True)
            logits = self.processors(input_ids, logits)
            entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
            if i > 0:
                entropy[k] -= eta
            k = entropy[1:].argmin() + 1
            logits_max = logits[k]
            logits_uncond = logits[0]
            logits_merged = (1 + beta) * logits_max - beta * logits_uncond
            logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
            # ========== NBCE ==========
            
            # tau = 0.01
            # probas = torch.nn.functional.softmax(logits[None] / tau , dim=-1)
            # next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1) 
            next_tokens = torch.argmax(logits, dim=-1).unsqueeze(-1)
            selected_tokens[i] = next_tokens[0].item()      
            if next_tokens[0] == self.tokenizer.eos_token_id:
                generated_len = i + 1
                break
            
            # prepare for next iteration
            input_ids = next_tokens.unsqueeze(-1).tile(batch_size, 1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device="cuda")], dim=-1)        

        return OutputAndCache(generated_len, selected_tokens.unsqueeze(0), past_key_values)


    def adjust_input_impl(self, accept_token_ids: torch.Tensor, 
                     proposer_input: InputAndCache, 
                     proposer_output: OutputAndCache) -> InputAndCache:
        proposer_input_ids = accept_token_ids.tile(proposer_input.input_ids.shape[0], 1)
        total_generated_len = proposer_output.past_key_values[0][0].shape[2] + 1
        proposer_key_values = crop_past_key_values(proposer_output.past_key_values, 
                                    max_len=total_generated_len - proposer_output.generated_len)
        logger.debug(f"adjust input: {total_generated_len}, {total_generated_len - proposer_output.generated_len}")
        proposer_attn_masks = torch.cat([proposer_input.attention_mask, 
                                         torch.ones_like(proposer_input_ids, dtype=torch.long)], dim=-1)
        return InputAndCache(proposer_input_ids, proposer_attn_masks, proposer_key_values)
    
    
class NBCEOptimizeProposer(NBCEProposer):
    def __init__(self, model, tokenizer, chunker) -> None:
        super().__init__(model, tokenizer, chunker)
    
    def set_prompt(self, prompts: List[str], past_key_values) -> InputAndCache:
        if len(prompts) > 1:
            raise NotImplementedError()
    
        inputs = self.tokenizer(prompts, padding='longest', return_tensors='pt').to('cuda')
        assert inputs.input_ids.shape[1] == past_key_values[0][0].shape[2]
        
        seq_len = inputs.input_ids.shape[-1]
        num_chunk = 8
        assert (seq_len % num_chunk) == 0
        chunk_size = seq_len // num_chunk
        
        input_ids = []
        for i in range(num_chunk):
            input_ids.append(inputs.input_ids[0, (i+1) * chunk_size - 1])
        input_ids = torch.tensor(input_ids, device='cuda').reshape(-1, 1)
        attention_mask = torch.cat([
            inputs.attention_mask.reshape(num_chunk, chunk_size),
            torch.ones((num_chunk, 1), dtype=torch.long, device='cuda')],
                                   dim = -1)
        # reshape kv cache
        new_past = []
        for idx in range(len(past_key_values)):
            bsz, num_head, seq_len, kv_dim = past_key_values[idx][0].shape
            new_past.append(
                (
                    past_key_values[idx][0].reshape(num_chunk * bsz, num_head, chunk_size, kv_dim),
                    past_key_values[idx][1].reshape(num_chunk * bsz, num_head, chunk_size, kv_dim)
                )
            )
        return InputAndCache(input_ids, attention_mask, tuple(new_past))