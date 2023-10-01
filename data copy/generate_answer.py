import json
from transformers import AutoTokenizer, LlamaForCausalLM
from fastchat.model.model_adapter import get_conversation_template
import torch
from tqdm import tqdm

model_path = "/data/vicuna-7b-v1.3/"
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
    
def generate_answer(prompt):
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt_with_template = conv.get_prompt()
    max_new_tokens = 128
    inputs = tokenizer([prompt_with_template], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:-1]
    generated_str = tokenizer.decode(generated)
    return generated_str


def main(filename):
    with open(filename) as f:
        data = json.load(f)

    for d in tqdm(data):
        assert len(d["conversation"]) == 1
        
        prompt = d["conversation"][0]["content"]
        answer = generate_answer(prompt)
        d["conversation"].append(
            {
                "role" : "assistant",
                "content" : answer
            }
        )
        
    with open(f"{filename.split('.')[0]}_with_answer.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    filename = "/home/lily/spec_new/data/spider_train.json"
    main(filename)