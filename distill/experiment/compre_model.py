import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from specInfer.generator import Generator
from train import LazySupervisedDataset


def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config).cuda()
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
                          tokenizer, max_propose_num)

    eval_json = json.load(open(data_path, "r"))
    eval_dataset = LazySupervisedDataset(eval_json, tokenizer=tokenizer,
                                         model=teacher_model_path, do_eval=True)

    i = 0
    correctness = 0
    stats = torch.zeros(32000, dtype=torch.long, device='cuda')
    alpha, sample_steps = 0, 0
    for d in eval_dataset:
        max_tokens = 512
        input_ids = d["input_ids"].reshape(1, -1).cuda()
        output = generator.generate(input_ids, max_tokens, temperature=1)
        correct_tokens = output.correct_tokens.squeeze(0)
        stats[correct_tokens] = stats[correct_tokens] + 1
        if i % 10 == 0:
            print(f"{i}/{len(eval_dataset)}")
        # print("===================================")
        # print(tokenizer.decode(d["input_ids"]))
        # print(output.output)
        # print(correct_tokens.shape)
        # print(output.propose_steps, correct_tokens.shape[-1]/output.propose_steps)
        # print("===================================")
        correctness += output.correct_tokens.shape[-1]/output.propose_steps
        alpha += output.alpha_sum
        sample_steps += output.sample_steps
        i += 1
        # if i == 1:
        #     break
    print(i, correctness / i, alpha.item() / sample_steps)

    # with open('output/spider.pkl', 'wb') as f:
    #     pickle.dump(stats, f)

def model_generate(model_path, data_path):
    tokenizer = AutoTokenizer.from_pretrained("/data/vicuna-7b-v1.3/")
    tokenizer.pad_token = tokenizer.unk_token
    model = load_model(model_path)
    param_sum = 0
    for param in model.parameters():
        param_sum += param.data.sum()
    print(f"Param Sum: {param_sum}")
    
    eval_json = json.load(open(data_path, "r"))
    eval_dataset = LazySupervisedDataset(eval_json, tokenizer=tokenizer,
                                         model=model_path, do_eval=True)

    i = 0
    for d in eval_dataset:
        max_tokens = 100
        input_ids = d["input_ids"].reshape(1, -1).cuda()
        generated = model.generate(input_ids,
                                   max_new_tokens=max_tokens)[
            0][input_ids.shape[-1]:]
        print(f"Prompt: {tokenizer.decode(input_ids[0])}")
        print(f"Answer: {tokenizer.decode(generated)}")
        print("----------------------------------")
        i += 1
        if i == 3:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str,
                        help="student model path",
                        default="eqhylxx/full-vicuna-160m")
    parser.add_argument("--teacher", type=str,
                        help="teacher model path",
                        default="/data/vicuna-7b-v1.3/")
    parser.add_argument("--data", type=str,
                        help="data path",
                        default="/home/lily/specNBCE/data/spider_eval.json")
    parser.add_argument("--max_propose_num", type=int,
                        help="number of proposed tokens",
                        default=5)

    args = parser.parse_args()
    main(args.student, args.teacher, args.max_propose_num, args.data)
    # model_generate(args.student, args.data)
