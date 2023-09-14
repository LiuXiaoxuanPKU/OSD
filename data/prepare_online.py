
import os
import json
import random

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)

def get_sample_idx(cases, i, total_size):
    def gen_prob(category_num, i):
        assert category_num == 3
        p1 = i / total_size * 0.7 + 0.1 # 0.1 ~ 0.8
        p2 = 0.8 - i / total_size * 0.7 # 0.8 ~ 0.1
        p3 = 1 - p1 - p2
        return [p1, p2, p3]      
        
    category_num = len(cases)
    probs = gen_prob(category_num, i)
    return random.choices(range(category_num), probs)[0]
    
def load_case(filename):
    cases = json.load(open(filename, "r"))
    return cases

def sample_cases(cases, total_size):
    sampled = []
    idx_in_category = [0 for _ in range(len(cases))]
    for i in range(total_size):
        category_id = get_sample_idx(cases, i, total_size)
        sample = cases[category_id][idx_in_category[category_id]]
        # print(sample)
        sampled.append(sample)
        idx_in_category[category_id] += 1
        idx_in_category[category_id] %= len(cases[category_id])
    return sampled
        
if __name__ == "__main__":
    datasets = ["spider", "arxiv-math", "mbpp"]
    cases = []
    for dataset in datasets:
        run_cmd(f"python clean_{dataset}.py")
        cases.append(load_case(f"{dataset}_train.json"))

    total_size = 5000
    sampled = sample_cases(cases, total_size)
    with open(f'smooth_mix.json', 'w') as f:
        json.dump(sampled, f)
        
    with open(f'sharp_mix.json', 'w') as f:
        json.dump(cases[0][:total_size//2] + cases[1][:total_size//2], f)
