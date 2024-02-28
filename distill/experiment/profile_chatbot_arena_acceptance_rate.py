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

from tqdm import tqdm
def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, device_map="cuda", torch_dtype=torch.bfloat16)
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
    for s in tqdm(range(len(eval_dataset)//30)):
        d = eval_dataset[s]

        alpha, sample_steps =0, 0
        alpha_i, sample_steps_i = 0, 0
        propose_steps_i, correct_count_i = 0, 0
        prompt_ids = d["input_ids"].reshape(1, -1).cuda()    
        eos_flag = False
        result = {'prompt': tokenizer.decode(prompt_ids[0], end="\n\n")}
        result['prompt_len'] =  len(prompt_ids[0])
        
        input_ids = prompt_ids
        iter_counter = 0
        gen_len = 0
        sd_records = []
        while not eos_flag:
            output = generator.generate(input_ids, 1, temperature=0.0001)

            correct_cnt = output.correct_tokens.shape[-1]
            propose_steps_i += output.propose_steps
            correct_count_i += correct_cnt

            alpha_i += output.alpha_sum
            sample_steps_i += output.sample_steps
            input_ids = torch.cat((input_ids, output.generated_ids[..., :1]), dim=-1)
            if tokenizer.eos_token_id in output.generated_ids:
                eos_flag = True
                gen_len += output.generated_ids.shape[-1]
            else:
                gen_len += 1

            record_i = {}
            record_i['gen_idx'] = iter_counter
            record_i['accepted_len'] = output.correct_tokens.shape[-1]
            if record_i['accepted_len'] == 10:
                print(output.correct_tokens)
                print(tokenizer.batch_decode(output.correct_tokens)[0])
                print(result['prompt'])

            record_i['confidences'], record_i['entropies'], record_i['first_appear'], record_i['appear_cnt'] = [], [], [], []
            record_i['target_prob'], record_i['target_max_prob'], record_i['target_entropy'], record_i['kl'] = [], [], [], []
            for conf_info in output.conf_infos:
                record_i['confidences'].append(conf_info.token_prob)
                record_i['entropies'].append(conf_info.token_entropy)
                correct_token_ids = (conf_info.next_token_id == conf_info.layer_token_ids).squeeze(0)
                if correct_token_ids.nonzero().numel() == 0:
                    record_i['first_appear'].append(-1)
                else:
                    record_i['first_appear'].append(correct_token_ids.nonzero()[0].item())
                record_i['appear_cnt'].append(sum(correct_token_ids).item())
                record_i['target_prob'].append(conf_info.target_prob)
                record_i['target_max_prob'].append(conf_info.target_max_prob)
                record_i['target_entropy'].append(conf_info.token_entropy)
                record_i['kl'].append(conf_info.kl_div)
            
            sd_records.append(record_i)
            iter_counter += 1

        result['gen_len'] = gen_len
        result['sd_records'] = sd_records

        result['avg_correct_count'] = correct_count_i / propose_steps_i
        result['avg_alpha'] = alpha_i.item()/sample_steps_i
        alpha_data.append(result)
        # print(result)
        print(gen_len, len(sd_records))

        correctness += correct_count_i / propose_steps_i
        alpha += alpha_i
        sample_steps += sample_steps_i

        i += 1

    # propose step: each step proposes a n-gram
    # sample step: single generation by student
    # correctness / i: average correct tokens per propose step (multiple tokens)
    # alpha / sample steps: average alpha per sample step (per propose)
    print(f'total data points: {i}, average correct tokens per propose step: {correctness / i}, average alpha per sample step: {alpha.item() / sample_steps}')

    with open('acc_T0_llama2_with_target.json', 'w') as f_write:
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
