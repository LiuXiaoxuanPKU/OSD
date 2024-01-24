from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from distill.specInfer.bench_generator import Generator
import json
from distill.train import LazySupervisedDataset
from tqdm import tqdm

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, device_map="cuda", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    return model


def main(student_model_path,
         teacher_model_path,
         max_propose_num,
         data_path):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    tokenizer.pad_token = tokenizer.unk_token
    teacher_model = load_model(teacher_model_path)
    student_model = load_model(student_model_path)

    generator = Generator(student_model, teacher_model,
                          tokenizer, max_propose_num, False)

    eval_json = json.load(open(data_path, "r"))
    eval_dataset = LazySupervisedDataset(eval_json, tokenizer=tokenizer,
                                         model=teacher_model_path, do_eval=True)


    for i in tqdm(range(len(eval_dataset)//30)):
        data = eval_dataset[i]
        prompt_ids = data["input_ids"].reshape(1, -1).cuda()
        data = []
        
        eos_flag = False
        input_ids = prompt_ids
        while not eos_flag:
            output = generator.generate(input_ids, 1, temperature=1.0)
            assert output.propose_steps == 1
            record = dict(
                prompt = TODO,
                prompt_token_ids = TODO,
                gen_token_ids = TODO,
                student_logits = TODO,
                teacher_logits = TODO,
                accepted_len = TODO
            )
            data.append(record)
            input_ids = torch.cat(TODO)
            
    torch.save(data, "data.pt")
    
if __name__ == "__main__":
    main()
