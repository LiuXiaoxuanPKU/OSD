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

    alpha_data = []

    i = 0
    correctness = 0
    vocab_size = len(tokenizer.get_vocab())
    stats = torch.zeros(vocab_size, dtype=torch.long, device='cuda')
    alpha, sample_steps = 0, 0
    unit_length = len(eval_dataset)//90
    for s in tqdm(range(unit_length, unit_length*2)):
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
        input_ids = prompt_ids
        eos_flag = False
        prompt_len = len(prompt_ids[0])
        result = {'prompt': tokenizer.decode(prompt_ids[0], end="\n\n")}
        result['prompt_len'] = prompt_len
        #len_to_rate_mapper = {}
        
        iter_counter = 0
        gen_len = prompt_len
        sd_records = []
        while not eos_flag:
            output = generator.generate(input_ids, max_tokens, temperature=0.001)

            correct_tokens = output.correct_tokens.squeeze(0)
            stats[correct_tokens] = stats[correct_tokens] + 1
            
            # sanity check for generation quality
            #if i % 10 == 0:
            #    print(f"{i}/{len(eval_dataset)}")
            #print("===================================")
            #print("Ref")
            #ref_generated = teacher_model.generate(input_ids, max_new_tokens=max_tokens)[0]
            #print(tokenizer.decode(ref_generated), end="\n\n")
            #print("--")
            #print(output.output[0])
            #print(correct_tokens.shape)
            #print(output.propose_steps, correct_tokens.shape[-1]/output.propose_steps)
            #print("===================================")

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

            #all_rates = {}
            #for c in range(1, max_tokens+1):
            #    if correct_cnt > c:
            #        rate = 1
            #    else:
            #        rate = correct_cnt / c
            #    all_rates[c] = rate

            #len_to_rate_mapper[prompt_len] = all_rates
            #print(len_to_rate_mapper[prompt_len])

            record_i = {}
            record_i['gen_idx'] = iter_counter
            record_i['accepted_len'] = correct_cnt

            prob_list = output.prob_list
            record_i['confidences'] = prob_list

            sd_records.append(record_i)

            gen_len += len(output.generated_ids)
            iter_counter += 1

        result['gen_len'] = gen_len
        result['sd_records'] = sd_records

        #result['prompt_length_to_alpha_map'] = len_to_rate_mapper

        result['avg_correct_count'] = correctness_i
        result['avg_alpha'] = alpha_i.item()/sample_steps_i
        alpha_data.append(result)
        print(result)

        correctness += avg_correctness_i
        alpha += alpha_i
        sample_steps += sample_steps_i

        i += 1

    # propose step: each step proposes a n-gram
    # sample step: single generation by student
    # correctness / i: average correct tokens per propose step (multiple tokens)
    # alpha / sample steps: average alpha per sample step (per propose)
    print(f'total data points: {i}, average correct tokens per propose step: {correctness / i}, average alpha per sample step: {alpha.item() / sample_steps}')

    with open('vicuna160m_chatbot_arena_all_token_acceptance_rate_for_simulation_temp_0p001_unit2.json', 'w') as f_write:
         json.dump(alpha_data, f_write, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str,
                        help="student model path",
                        default="models/vicuna-160m")
    parser.add_argument("--teacher", type=str,
                        help="teacher model path",
                        default="models/vicuna-7b-v1.3")
    parser.add_argument("--data", type=str,
                        help="data path",
                        default="data/raw_data/chatbot_arena_token_acceptance_rate_testing.json")
    parser.add_argument("--max_propose_num", type=int,
                        help="number of proposed tokens",
                        default=10)

    args = parser.parse_args()
    main(args.student, args.teacher, args.max_propose_num, args.data)
