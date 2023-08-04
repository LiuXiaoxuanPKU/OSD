
from transformers import TopPLogitsWarper, LogitsProcessorList
import torch
from dataclasses import dataclass
from typing import Tuple, List
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from chunker import DummyChunker, LongchatChunker

import logging
logger = logging.getLogger('generator_logger') 
logger.setLevel(logging.INFO) 
handler = logging.FileHandler('generator.log')
handler.setLevel(logging.DEBUG)  # Set the minimum log level for this handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass
class InputAndCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: torch.Tensor

@dataclass
class OutputAndCache:
    generated_len: int
    output_ids: torch.Tensor
    past_key_values: torch.Tensor
    
class NBCEProposer:
    def __init__(self, model, tokenizer, chunker) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.processors = LogitsProcessorList()
        self.processors.append(TopPLogitsWarper(0.95))
    
    def get_input(self, prompts: List[str]) -> InputAndCache:
        if len(prompts) > 1:
            raise NotImplementedError()
        prompt = prompts[0]
    
        chunked_prompts = self.chunker.chunk(prompt)
        inputs = self.tokenizer([''] + chunked_prompts, padding='longest', return_tensors='pt').to('cuda')
        
        return InputAndCache(inputs.input_ids, inputs.attention_mask, None)
    
    def propose(self, input: InputAndCache, n: int):
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

class Verifier:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_inputs = None
        self.processor = LogitsProcessorList()
    
    def get_input(self, prompts) -> InputAndCache:
        self.prompt_inputs = self.tokenizer(prompts, padding="longest", return_tensors="pt").to(device="cuda")
        logger.debug(f"prompt input length: {self.prompt_inputs.input_ids.shape}")
        return InputAndCache(self.prompt_inputs.input_ids, self.prompt_inputs.attention_mask, None)
    
    def verify(self, input: InputAndCache, max_propose_tokens: int) -> Tuple[InputAndCache, torch.Tensor]:
        outputs = self.model(input_ids=input.input_ids, 
                             attention_mask=input.attention_mask, 
                             past_key_values=input.past_key_values)
        next_token_scores = self.processor(input.input_ids, outputs.logits)
        generated_len = max_propose_tokens + 1
        next_tokens = torch.argmax(next_token_scores[:, -generated_len:, :], dim=-1)
        return OutputAndCache(generated_len, next_tokens, outputs.past_key_values)
             
class Generator:
    def __init__(self, model, tokenizer, chunker) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.proposer = NBCEProposer(model, tokenizer, chunker)
        self.verifier = Verifier(model, tokenizer)
        
        # parameters
        self.max_propose_tokens = 16
        
    def prepare_verifier_input(self, proposer_output: OutputAndCache, 
                                    verifier_input: InputAndCache,
                                    max_propose_tokens: int) -> InputAndCache:
        logger.debug(proposer_output.output_ids.shape)
        if verifier_input.past_key_values is None:
            # concatenate proposed inputs with prompts
            input_ids = torch.cat([self.verifier.prompt_inputs.input_ids, proposer_output.output_ids], dim=-1)
            # prompt phase, we don't have kv cache (past_key_values)
            past_key_values = None
            # concatenate prompt masks with proposed token masks
            attention_mask = torch.cat([self.verifier.prompt_inputs.attention_mask, 
                                        torch.ones_like(proposer_output.output_ids, 
                                                        dtype=torch.long, device="cuda")], dim=-1)
        else:
            input_ids = torch.cat([verifier_input.input_ids, proposer_output.output_ids], dim=-1)
            past_key_values = verifier_input.past_key_values
            attention_mask = torch.cat([verifier_input.attention_mask, 
                                        torch.ones_like(proposer_output.output_ids, 
                                                        dtype=torch.long, device="cuda")], dim=-1)  
        return InputAndCache(input_ids, attention_mask, past_key_values)
    
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

    def adjust_inputs(self, accept_token_ids: torch.Tensor, max_propose_tokens: int,
                      proposer_input: InputAndCache, verifier_input: InputAndCache,
                      proposer_output: OutputAndCache, verifier_output: OutputAndCache) -> Tuple[InputAndCache, InputAndCache]:
        def _crop_past_key_values(past_key_values, max_len):
            new_past = []
            for idx in range(len(past_key_values)):
                new_past.append(
                (
                    past_key_values[idx][0][:, :, :max_len, :],
                    past_key_values[idx][1][:, :, :max_len, :],
                )
                )
            return tuple(new_past)
        
        proposer_input_ids = accept_token_ids.tile(proposer_input.input_ids.shape[0], 1)
        total_generated_len = proposer_output.past_key_values[0][0].shape[2] + 1
        proposer_key_values = _crop_past_key_values(proposer_output.past_key_values, 
                                    max_len=total_generated_len - proposer_output.generated_len)
        logger.debug(f"adjust input: {total_generated_len}, {total_generated_len - proposer_output.generated_len}")
        proposer_attn_masks = torch.cat([proposer_input.attention_mask, 
                                         torch.ones_like(proposer_input_ids, dtype=torch.long)], dim=-1)
        
        n_matches = accept_token_ids.shape[1]
        verifier_input_ids = verifier_output.output_ids[:, n_matches-1:n_matches]
        verifier_generated_len = verifier_output.past_key_values[0][0].shape[2] - max_propose_tokens + n_matches
        verifier_key_values = _crop_past_key_values(verifier_output.past_key_values, verifier_generated_len - 1)
        
        verifier_attn_masks = verifier_input.attention_mask[:, :verifier_generated_len]
        if verifier_attn_masks.shape[1] < verifier_generated_len:
            verifier_attn_masks = torch.cat([verifier_attn_masks, 
                                        torch.ones(verifier_attn_masks.shape[0], 1, dtype=torch.long, device="cuda")], dim=-1)  
        
        return (InputAndCache(proposer_input_ids, proposer_attn_masks, proposer_key_values), 
                InputAndCache(verifier_input_ids, verifier_attn_masks, verifier_key_values))

    @torch.inference_mode()
    def generate(self, batch, max_tokens):
        proposer_input = self.proposer.get_input(batch)
        verifier_input = self.verifier.get_input(batch)
        
        generated_token_cnt = 0
        generated_tokens = None
        while True:
            # propose n tokens
            proposer_output = self.proposer.propose(proposer_input, self.max_propose_tokens)
            
            # prepare verifier input
            verifier_input = self.prepare_verifier_input(proposer_output, 
                                                         verifier_input, 
                                                         self.max_propose_tokens)
            
            # forward n tokens on the model in the a single batch
            verifier_output = self.verifier.verify(verifier_input, self.max_propose_tokens)
            
            # compare selected tokens
            accept_token_ids = self.compare_tokens(proposer_output, verifier_output)
            logger.info(accept_token_ids.shape)
            if generated_tokens is None:
                generated_tokens = accept_token_ids
            else:
                generated_tokens = torch.cat([generated_tokens, accept_token_ids], dim=-1)
            generated_token_cnt += accept_token_ids.shape[1]
            
            if generated_token_cnt >= max_tokens or self.tokenizer.eos_token_id in accept_token_ids:
                break
            
            # adjust the proposer/verifier input, discard unnecessary kv cache
            proposer_input, verifier_input = self.adjust_inputs(accept_token_ids, self.max_propose_tokens,
                                                                proposer_input, verifier_input,
                                                                proposer_output, verifier_output)
            
            logger.debug("================================")
        logger.debug(generated_tokens)
        return self.tokenizer.batch_decode(generated_tokens)


if __name__ == "__main__":
    # model_path = "/rscratch/zhendong/lily/longchat-7b-16k/"
    model_path = "/rscratch/zhendong/lily/vicuna-7b-v1.3/"
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    chunker = DummyChunker()
    generator = Generator(model, tokenizer, chunker)
    prompt = ["Give a five day hawaii travel plan" ]
    generated = generator.generate(prompt, 100)
    logger.debug(f"generated: {generated}")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    ref_generated = model.generate(**inputs, max_new_tokens=100)
    logger.debug(ref_generated)
    logger.debug(f"ref_generated: {tokenizer.batch_decode(ref_generated)}")
    
    
    
    