import json
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForSeq2SeqLM
from fastchat.model.model_adapter import get_conversation_template
import torch
from tqdm import tqdm

import datasets

import os

model_path = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def generate_answer_gsm8k(prompt):
    prefix = "Q: Solve the following math problem. "
    prompt = prefix + prompt
    max_new_tokens = 128
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    # generate, remove bos and eos
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][1:-1]
    generated_str = tokenizer.decode(generated)
    return prompt, generated_str


def main(filename):
    trainset = datasets.Dataset.from_list(read_jsonl(filename))

    def mapping(d):
        question = d.pop('question')
        prompt, answer = generate_answer_gsm8k(question)
        
        d['question'] = prompt
        d['answer'] = answer

        return d
    
    trainset = trainset.map(mapping)
    
    path = "/home/lanxiang/MIT/LLMs_and_TVM/specd/specNBCE-main/data/"
    new_path = os.path.join(path, f"gsm8k_with_answer_t5.jsonl")
    print(new_path)
    trainset.to_json(new_path)
    
    #with open(f"{filename.split('.')[0]}_with_answer_t5.json", "w") as f:
    #    json.dump(trainset, f)

if __name__ == "__main__":
    filename = "/home/lanxiang/MIT/LLMs_and_TVM/specd/specNBCE-main/data/gsm8k/train.jsonl"
    main(filename)