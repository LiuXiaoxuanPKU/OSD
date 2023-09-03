import argparse
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from specInfer.generator import Generator
from train import LazySupervisedDataset
import pickle
import torch

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config).cuda()
    return model

def main(student_model_path, teacher_model_path,
         data_path):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    tokenizer.pad_token = tokenizer.unk_token
    teacher_model = load_model(teacher_model_path)
    student_model = load_model(student_model_path)
    generator = Generator(student_model, teacher_model, tokenizer)
   
    eval_json = json.load(open(data_path, "r"))
    eval_dataset = LazySupervisedDataset(eval_json, tokenizer=tokenizer, 
                                         model=teacher_model_path, do_eval=True)
    
    
    i = 0
    correctness = 0
    stats = torch.zeros(32000, dtype=torch.long, device='cuda')
    for d in eval_dataset:
        max_tokens = 100
        input_ids = d["input_ids"].reshape(1, -1).cuda()
        output, correct_tokens, propose_step = generator.generate(input_ids, max_tokens)
        correct_tokens = correct_tokens.squeeze(0)
        stats[correct_tokens] = stats[correct_tokens] + 1
        if i % 10 == 0:
            print(f"{i}/{len(eval_dataset)}")
        print("===================================")
        print(tokenizer.decode(d["input_ids"]))
        print(output)
        print(correct_tokens.shape)
        print(propose_step, correct_tokens.shape[-1]/propose_step)
        print(tokenizer.batch_decode(correct_tokens))
        correctness += correct_tokens.shape[-1]/propose_step
        print("===================================")
        i += 1
        if i == 10:
            exit(0)
    print(i, correctness / i)
    
    with open('output/vicuna-7k_correct.pkl', 'wb') as f:
        pickle.dump(stats, f)


def model_generate(model_path, data_path):
    tokenizer = AutoTokenizer.from_pretrained("/data/vicuna-7b-v1.3/")
    tokenizer.pad_token = tokenizer.unk_token
    model = load_model(model_path)
    eval_json = json.load(open(data_path, "r"))
    eval_dataset = LazySupervisedDataset(eval_json, tokenizer=tokenizer, 
                                         model=model_path, do_eval=True)
    
    i = 0
    for d in eval_dataset:
        max_tokens = 100
        input_ids = d["input_ids"].reshape(1, -1).cuda()
        generated = model.generate(input_ids, max_new_tokens=max_tokens)[0][input_ids.shape[-1]:]
        print(f"Prompt: {tokenizer.decode(input_ids[0])}")
        print(f"Answer: {tokenizer.decode(generated)}")
        print("----------------------------------")
        i += 1
        if i == 10:
            break
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, 
                        help="student model path", 
                        default="eqhylxx/vicuna-160m")
    parser.add_argument("--teacher", type=str, 
                        help="teacher model path", 
                        default="/data/vicuna-7b-v1.3/")
    # parser.add_argument("--teacher", type=str, 
    #                     help="teacher model path", 
    #                     default="/data/vicuna_cip/checkpoint-7000")
    
    parser.add_argument("--data", type=str, 
                        help="data path", 
                        default="/home/lily/specNBCE/data/spider_eval.json")
    
    args = parser.parse_args()
    # main(args.student, args.teacher, args.data)
    model_generate(args.student, args.data)