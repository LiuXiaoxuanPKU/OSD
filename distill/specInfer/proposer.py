import torch
from typing import List
from specInfer.common import (InputAndCache, 
                              OutputAndCache, 
                              crop_past_key_values, 
                              sychronize_time,
                              sample_fn)

class Proposer:
    def __init__(self) -> None:
        self.propose_time = 0
        self.adjust_time = 0
        
        self.benchmark_time = False
    
    def propose(self, input: InputAndCache, n: int) -> OutputAndCache:
        if self.benchmark_time:
            start = sychronize_time()
            
        ret = self.propose_impl(input, n)
        
        if self.benchmark_time:
            self.propose_time += sychronize_time() - start
        
        return ret
    
    def adjust_input(self, accept_token_ids: torch.Tensor, 
                     proposer_input: InputAndCache, 
                     proposer_output: OutputAndCache) -> InputAndCache:
        if self.benchmark_time:
            start = sychronize_time()
        
        ret = self.adjust_input_impl(accept_token_ids, proposer_input, proposer_output)
        
        if self.benchmark_time:
            self.adjust_time += sychronize_time() - start
        
        return ret
    
    def __del__(self):
        if self.benchmark_time:
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

class SmallModelProposer(Proposer):
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def propose_impl(self, input: InputAndCache, n: int) -> OutputAndCache:
        if input.input_ids.shape[0] > 1:
            raise NotImplementedError("Not implement for batch_size > 1 in evaluation")
        
        propose_tokens = []
        input_ids = input.input_ids
        generated_len = n
        for i in range(n):
            outputs = self.model(input_ids)
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            propose_tokens.append(next_tokens[0].item())
            if next_tokens[0] == self.tokenizer.eos_token_id:
                generated_len = i + 1
                break
            input_ids = torch.cat([input_ids, next_tokens.reshape(1, 1)], dim=-1)
        propose_tokens = torch.tensor(propose_tokens, device=input_ids.device).reshape(1, -1)
        return OutputAndCache(generated_len, propose_tokens, None, None)
    
    def adjust_input_impl(self, accept_token_ids: torch.Tensor, 
                     proposer_input: InputAndCache, 
                     proposer_output: OutputAndCache) -> InputAndCache:
        new_input_ids = torch.cat([proposer_input.input_ids, accept_token_ids], dim=-1)
        return InputAndCache(new_input_ids, torch.ones_like(new_input_ids), None)
    
class SmallModelKVCacheProposer(Proposer):
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def propose_impl(self, input: InputAndCache, n: int) -> OutputAndCache:
        if input.input_ids.shape[0] > 1:
            raise NotImplementedError("Not implement for batch_size > 1 in evaluation")
        
        propose_tokens = []
        propose_logits = []
        input_ids = input.input_ids
        past_key_values = input.past_key_values
        generated_len = n
        for i in range(n):
            outputs = self.model(input_ids,
                                 past_key_values=past_key_values,
                                 use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = sample_fn(next_token_logits).item()
            propose_tokens.append(next_token_id)
            if next_token_id == self.tokenizer.eos_token_id:
                generated_len = i + 1
                print(f"[Info] Stop at {generated_len} because of eos")
                break
            input_ids = torch.tensor(next_token_id, device='cuda', dtype=torch.long).reshape(1, -1)
            propose_logits.append(outputs.logits[:, -1, :])
        propose_tokens = torch.tensor(propose_tokens, device=input_ids.device).reshape(1, -1)
        propose_logits = torch.cat(propose_logits, dim=0)
        return OutputAndCache(generated_len, propose_tokens, propose_logits, past_key_values)
    
    def adjust_input_impl(self, accept_token_ids: torch.Tensor, 
                     proposer_input: InputAndCache, 
                     proposer_output: OutputAndCache) -> InputAndCache:
        proposer_input_ids = accept_token_ids.tile(proposer_input.input_ids.shape[0], 1)
        total_generated_len = proposer_output.past_key_values[0][0].shape[2] + 1
        proposer_key_values = crop_past_key_values(proposer_output.past_key_values, 
                                    max_len=total_generated_len - proposer_output.generated_len)
        proposer_attn_masks = torch.cat([proposer_input.attention_mask, 
                                         torch.ones_like(proposer_input_ids, dtype=torch.long)], dim=-1)
        return InputAndCache(proposer_input_ids, proposer_attn_masks, proposer_key_values)