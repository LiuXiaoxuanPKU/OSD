import torch
from transformers import (AutoTokenizer, LlamaForCausalLM)
from chunker import DummyChunker, LongchatChunker
from generator import Generator

PROMPTS = [
    "Give a five day hawaii travel plan",
    "What's your name, do you like hiking?",
    "dummy\n" * 100 + "what's your name, Do you want to go hiking?"
]

def test_end2end_proposer_verifier():
    model_path = "/rscratch/zhendong/lily/vicuna-7b-v1.3/"
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    chunker = DummyChunker()
    generator = Generator(model, tokenizer, chunker)
    correct = 0
    for prompt in PROMPTS:
        prompt = [prompt]
        generated = generator.generate(prompt, 100)[0]
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        ref_generated = model.generate(**inputs, max_new_tokens=100)
    
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        ref_generated = tokenizer.batch_decode(model.generate(**inputs, max_new_tokens=100))[0][len(prompt[0]) + 1:]
        if ref_generated not in generated:
            print("==============================")
            print(f"[Error] {generated}\n---------\n{ref_generated}")
            print("==============================")
        else:
            correct += 1
    print(f"[Success] {correct} / {len(PROMPTS)}")

if __name__ == "__main__":
    test_end2end_proposer_verifier()