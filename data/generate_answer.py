import json
from transformers import AutoTokenizer, LlamaForCausalLM
from fastchat.model.model_adapter import get_conversation_template

def generate_answer(prompt):
    model_path = "/rscratch/zhendong/lily/vicuna-7b-v1.3/"
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], prompt)
    prompt_with_template = conv.get_prompt()
    
    max_new_tokens = 128
    inputs = tokenizer([prompt_with_template], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][inputs.input_ids.shape[-1]:]
    generated_str = tokenizer.decode(generated)
    print(generated_str)
    exit(0)


def main(filename):
    with open(filename) as f:
        data = json.load(f)

    for d in data:
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
        f.dump(data)

if __name__ == "__main__":
    filename = ""
    main(filename)