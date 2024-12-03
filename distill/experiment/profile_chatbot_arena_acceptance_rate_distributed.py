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
from specInfer.bench_generator import Generator
from train import LazySupervisedDataset

from accelerate import init_empty_weights, infer_auto_device_map

from tqdm import tqdm

from pprint import pprint

def load_model_distributed(model_path):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()

    # Infer device map for the model
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "46GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["LlamaDecoderLayer"]  # Prevent splitting decoder layers
    )

    # Optional: Print the device map to verify layer assignments
    pprint(device_map)

    # Load the model across GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    return model

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, device_map="cuda:5", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    return model

def main(student_model_path,
         teacher_model_path,
         max_propose_num,
         data_path):
    
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = load_model_distributed(teacher_model_path)
    student_model = load_model_distributed(student_model_path)

    generator = Generator(student_model, teacher_model,
                          tokenizer, max_propose_num, False)

    eval_json = json.load(open(data_path, "r"))
    eval_dataset = LazySupervisedDataset(eval_json, tokenizer=tokenizer,
                                         model=teacher_model_path, do_eval=True)
    
    print("eval dataset size:")
    print(len(eval_dataset))

    alpha_data = []

    i = 0
    correctness = 0
    vocab_size = len(tokenizer.get_vocab())
    stats = torch.zeros(vocab_size, dtype=torch.long, device='cuda')
    alpha, sample_steps = 0, 0
    unit_length = len(eval_dataset)//150 # FIXME (lanxiang): make this configurable
    for s in tqdm(range(unit_length)):
        d = eval_dataset[s]
        if i % 10 == 0:
            print(f"data: {i}/{len(eval_dataset)}")
        max_tokens = 10
        correctness_i = 0
        avg_correctness_i = 0
        correct_count_i = 0
        alpha_i, sample_steps_i = 0, 0
        propose_steps_i = 0
        prompt_ids = d["input_ids"].reshape(1, -1).cuda()
        print("prompt id length:")
        print(prompt_ids.shape[-1])
        if prompt_ids.shape[-1] > 512:
            print('skpping due to super long prompt > 512...')
            continue
        input_ids = prompt_ids
        eos_flag = False
        prompt_len = len(prompt_ids[0])
        result = {'prompt': tokenizer.decode(prompt_ids[0], end="\n\n")}
        result['prompt_len'] = prompt_len
        
        iter_counter = 0
        gen_len = prompt_len
        sd_records = []
        while not eos_flag:
            output = generator.generate(input_ids, max_tokens, temperature=0.001)

            correct_tokens = output.correct_tokens.squeeze(0)
            stats[correct_tokens] = stats[correct_tokens] + 1

            correct_cnt = output.correct_tokens.shape[-1]
            propose_steps_i += output.propose_steps
            correct_count_i += correct_cnt
            correctness_i = correct_count_i/propose_steps_i
            avg_correctness_i = correctness_i

            alpha_i += output.alpha_sum
            sample_steps_i += output.sample_steps
            input_ids = torch.cat((input_ids, output.generated_ids[..., :1]), dim=-1)
            if tokenizer.eos_token_id in output.generated_ids:
                eos_flag = True

            prompt_len = input_ids.shape[-1]

            record_i = {}
            record_i['gen_idx'] = iter_counter
            record_i['accepted_len'] = correct_cnt

            prob_list = output.prob_list
            record_i['confidences'] = prob_list

            sd_records.append(record_i)

            gen_len += 1
            iter_counter += 1

        result['gen_len'] = gen_len
        result['sd_records'] = sd_records

        result['avg_correct_count'] = correctness_i
        result['avg_alpha'] = alpha_i.item()/sample_steps_i
        alpha_data.append(result)
        print(result)

        correctness += avg_correctness_i
        alpha += alpha_i
        sample_steps += sample_steps_i

        i += 1

    print(f'total data points: {i}, average correct tokens per propose step: {correctness / i}, average alpha per sample step: {alpha.item() / sample_steps}')

    with open('llama3.2_1b_chatbot_arena_all_token_acceptance_rate_01.json', 'w') as f_write:
         json.dump(alpha_data, f_write, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str,
                        help="student model path",
                        default="/home/models/Llama-3.2-1B-Instruct")
    parser.add_argument("--teacher", type=str,
                        help="teacher model path",
                        default="/home/models/Llama-3.1-70B-Instruct")
    parser.add_argument("--data", type=str,
                        help="data path",
                        default="/workspace/workspace/OSD/data/raw_data/chatbot_arena_token_acceptance_rate_testing.json")
    parser.add_argument("--max_propose_num", type=int,
                        help="number of proposed tokens",
                        default=10)

    args = parser.parse_args()
    main(args.student, args.teacher, args.max_propose_num, args.data)