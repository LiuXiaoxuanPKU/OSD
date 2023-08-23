import torch
import time
from transformers import AutoTokenizer, LlamaForCausalLM
from chunker import DummyChunker
from common import OutputAndCache
from proposer import RandomProposer, NBCEProposer, NBCEOptimizeProposer
from verifier import Verifier, OptimizeVerifier
from common import sychronize_time

import logging
logger = logging.getLogger('generator_logger') 
logger.setLevel(logging.WARNING) 
handler = logging.FileHandler('generator.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

             
class Generator:
    def __init__(self, model, tokenizer, chunker, proposer=None, verifier=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.proposer = NBCEProposer(model, tokenizer, chunker) if proposer is None else proposer
        # self.proposer = RandomProposer()
        self.verifier = Verifier(model, tokenizer) if verifier is None else verifier
        
        # parameters
        self.max_propose_tokens = 16
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
        # perfect match
        n_matches = proposed_output.output_ids.shape[-1]
        return verified_output.output_ids[:, :n_matches + 1]

    @torch.inference_mode()
    def generate(self, batch, max_tokens):
        start = sychronize_time()
        verifier_input = self.verifier.set_prompt(batch)
        proposer_input = self.proposer.set_prompt(batch, verifier_input.past_key_values)
        prompt_time = sychronize_time() - start
        self.generation_time.append(prompt_time)
        
        generated_token_cnt = 0
        generated_tokens = None
        while True:
            start = sychronize_time()
            # propose n tokens
            proposer_output = self.proposer.propose(proposer_input, self.max_propose_tokens)
            
            # prepare verifier input
            verifier_input = self.verifier.prepare_input(proposer_output, 
                                                         verifier_input)
            
            # forward n tokens on the model in the a single run
            verifier_output = self.verifier.verify(verifier_input, self.max_propose_tokens)
            
            # compare selected tokens
            accept_token_ids = self.compare_tokens(proposer_output, verifier_output)
            logger.info(accept_token_ids.shape)
            if generated_tokens is None:
                generated_tokens = accept_token_ids
            else:
                generated_tokens = torch.cat([generated_tokens, accept_token_ids], dim=-1)
            generated_token_cnt += accept_token_ids.shape[1]
            
            
            # adjust the proposer/verifier input, discard unnecessary kv cache
            proposer_input = self.proposer.adjust_input(accept_token_ids, proposer_input, proposer_output)
            verifier_input = self.verifier.adjust_input(accept_token_ids, verifier_input, verifier_output)
            
            self.generation_time.append(sychronize_time() - start)
            
            # if generated_token_cnt >= max_tokens or self.tokenizer.eos_token_id in accept_token_ids:
            #     break
            
            if generated_token_cnt >= max_tokens:
                break
            logger.debug("================================")
        logger.debug(generated_tokens)
        logger.info(f"generated tokens: {generated_tokens.shape}")
        return self.tokenizer.batch_decode(generated_tokens)
    
    def __del__(self):
        print(f"[Generator time: {self.generation_time}")
        print(f"[Max allocated memory]: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")


if __name__ == "__main__":
    model_path = "/data/longchat-7b-16k/"
    # model_path = "/rscratch/zhendong/lily/vicuna-7b-v1.3/"
    # model_path = "facebook/opt-125m"
    from monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense
    from monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_with_condense()
    replace_llama_attn_with_flash_attn()
    
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    chunker = DummyChunker()
    generator = Generator(model, tokenizer, chunker, 
                          NBCEOptimizeProposer(model, tokenizer, chunker),
                          OptimizeVerifier(model, tokenizer))
    prompt = ("Could you tell me the main idea of the following paragraph: " +
            "Motivation is useful for activities that are considered dull" + 
            "(e.g., washing the dishes), whereas passion is the driving force " +
            "for activities that have significance for us. " +
            "Passion can be negative or positive, however. " +
            "Negative passions, referred to as obsessive passions, " + 
            "are maladaptive and lead to unhealthy behaviors; these types of " +
            "passions should be avoided. On the other hand, positive, harmonious ")
    prompts = [prompt]
    # warmup
    ref_generated = []
    start = time.time()
    for prompt in prompts:
        inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
        ref_generated.append(model.generate(**inputs, max_new_tokens=100)[0][inputs.input_ids.shape[-1]:])
    print(f"time: {time.time() - start}, ref_generated:",
          f"{tokenizer.batch_decode(ref_generated)}")
    
    # generated = []
    # start = time.time()
    # for prompt in prompts:
    #     generated.append(generator.generate([prompt], 100)[0])
    # print(f"time: {time.time() - start}, generated: {generated}")
    
    # ref_generated = []
    # start = time.time()
    # for prompt in prompts:
    #     inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
    #     ref_generated.append(model.generate(**inputs, max_new_tokens=100)[0])
    # print(f"time: {time.time() - start}, ref_generated: {tokenizer.batch_decode(ref_generated)}")
    
    
    
    