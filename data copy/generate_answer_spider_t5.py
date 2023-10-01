import json
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForSeq2SeqLM
from fastchat.model.model_adapter import get_conversation_template
import torch
from tqdm import tqdm
import os

import datasets

import os

model_path = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda:1")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_answer_spider(prompt):
    prefix = "Q: translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "
    prompt = prefix + prompt
    max_new_tokens = 128
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][1:-1]
    generated_str = tokenizer.decode(generated)
    return prompt, generated_str

def main(dataset_name):
    trainset = datasets.load_dataset(dataset_name,)['train']

    def mapping(d):
        question = d.pop('question')
        prompt, answer = generate_answer_spider(question)
        
        d['question'] = prompt
        d['query'] = answer

        return d
    
    trainset = trainset.map(mapping)
    
    path = "/home/lanxiang/MIT/LLMs_and_TVM/specd/specNBCE-main/data/"
    new_path = os.path.join(path, f"{dataset_name}_with_answer_t5.jsonl")
    print(new_path)
    trainset.to_json(new_path)
    #with open(f"{dataset_name.split('_')[0]}_with_answer_t5.json", "w") as f:
    #    json.dump(trainset, f)

if __name__ == "__main__":
    dataset_name = 'spider'
    main(dataset_name)