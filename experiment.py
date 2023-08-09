from transformers import AutoTokenizer, LlamaForCausalLM 
import torch
from common import slice_past_key_values, crop_past_key_values
from fastchat.model import get_conversation_template, load_model
import argparse

@torch.inference_mode()
def model_generate(model, input_ids, att_mask, past_key_values, position_ids=None):
    outputs = model(input_ids=input_ids, 
                        attention_mask=att_mask, 
                        past_key_values=past_key_values,
                        position_ids=position_ids)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    return next_token, tokenizer.batch_decode([next_token])[0], outputs.past_key_values   

def chunk_exp(model, tokenizer, prompt):
    chunk_generated = {}
    max_new_tokens = 20
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
    input_ids, attention_mask, past_key_values = inputs.input_ids, inputs.attention_mask, None
    prompt_len = input_ids.shape[1]
    num_chunk = 4
    chunk_size = prompt_len//num_chunk
    print("==========prompt=========")
    print(prompt)
    print("=========================")
    for token_id in range(max_new_tokens):
        next_token, ref_token, past_key_values = model_generate(model, 
                                                                input_ids,
                                                                attention_mask, 
                                                                past_key_values)
        # print(past_key_values[0][0].shape)
        next_token = next_token.unsqueeze(1)
        chunk_attn_mask = torch.ones((1, chunk_size + 1 + token_id))
        if len(chunk_generated) == 0:
            for start_idx in range(num_chunk):
                chunk_generated[start_idx] = [next_token.squeeze(0)]
        for start_idx in range(num_chunk):
            chunk_past_key_values = slice_past_key_values(past_key_values, start_idx * chunk_size, chunk_size)
            # position_start_id = (start_idx + 1) * chunk_size + token_id
            # position_start_id =  chunk_past_key_values[0][0].shape[2]
            # chunk_position_ids = torch.arange(start=position_start_id, end=position_start_id+token_id+1, step=1, device='cuda', requires_grad=False, dtype=torch.long)
            # chunk_position_ids = chunk_position_ids.unsqueeze(0)
            chunk_position_ids = None
            chunk_input_ids = torch.cat(chunk_generated[start_idx], dim=-1).unsqueeze(0)
            chunk_token_id, chunk_token, _ = model_generate(model, 
                                               chunk_input_ids, 
                                               chunk_attn_mask, 
                                               chunk_past_key_values,
                                               chunk_position_ids)
            chunk_generated[start_idx].append(chunk_token_id)
        # prepare for next iteration
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long, device="cuda")], dim=-1)        
    print("==========chunk generated=========")
    for k in chunk_generated:
        print(k, [c.item() for c in chunk_generated[k]])
        print(k, [tokenizer.decode(c.item()) for c in chunk_generated[k]])
    print("==========ref generated=========")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)[0][prompt_len:]
    print(out)
    print(tokenizer.decode(out))

def mask_exp(model, tokenizer, prompt):
    def mask_lower(mask):
        bs, kv_len = mask.shape[0], mask.shape[1]
        for i in range(bs):
            for j in range(0, kv_len//2):
                mask[i, j] = 0
        return mask
    
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
    max_new_tokens = 20
    
    prompt_len = inputs.input_ids.shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, 
                         output_scores=True,
                         return_dict_in_generate=True)
    # for idx, logit in enumerate(out.scores):
    #     print(idx, logit.max().item(), logit.argmax(-1))
    print("=============Ref Output=================")
    print(tokenizer.decode(out[0][0]))
    
    # attention_mask = mask_lower(inputs.attention_mask)
    attention_mask = inputs.attention_mask
            
    max_new_tokens = 20
    past_key_values = None
    input_ids = inputs.input_ids
    print("=============Masked Output=================")
    for i in range(max_new_tokens):
        next_token_id, next_token, past_key_values = model_generate(model, 
                                                                input_ids,
                                                                attention_mask, 
                                                                past_key_values)
        seq_len = past_key_values[0][0].shape[2] + 1
        input_ids = next_token_id.unsqueeze(1)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device="cuda") 
        attention_mask = mask_lower(attention_mask)
        print(next_token, end='')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path", default="/rscratch/zhendong/lily/longchat-7b-16k/")
    args = parser.parse_args()
    
    model_path = args.model
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)

    model, tokenizer = load_model(
            model_path,
            device="cuda",
            num_gpus=1,
            max_gpu_memory=f"48GiB",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = ("Could you tell me the main idea of the following paragraph: " +
            "Motivation is useful for activities that are considered dull" + 
            "(e.g., washing the dishes), whereas passion is the driving force " +
            "for activities that have significance for us. " +
            "Passion can be negative or positive, however. " +
            "Negative passions, referred to as obsessive passions, " + 
            "are maladaptive and lead to unhealthy behaviors; these types of " +
            "passions should be avoided. On the other hand, positive, harmonious ")
    
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_with_templates = conv.get_prompt()
       
    # chunk_exp(model, tokenizer, prompt_with_templates)
    mask_exp(model, tokenizer, prompt_with_templates)     

   