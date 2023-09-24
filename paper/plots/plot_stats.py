import pickle as pk
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import seaborn as sns

with open('ckpt500-spider.pkl', 'rb') as file:
    stats = pk.load(file)

ckpt_stats, stats = stats[0], stats[1]

with open('org-spider.pkl', 'rb') as file:
    org_stats, _ = pk.load(file)
    
k = 50
non_zero_indexes = torch.nonzero(stats, as_tuple=True)
indexes = non_zero_indexes[0].tolist()
values = stats[non_zero_indexes].tolist()
total_occ = sum(values)

pairs = zip(indexes, values)
sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
top_occr = sum([x[1] for x in sorted_pairs[:k]])
print(top_occr * 1.0 / total_occ)

top_tokens = [x[0] for x in sorted_pairs[:k]]
ckpt_acc = [ckpt_stats[t].item() / stats[t].item() for t in top_tokens][1:]
org_acc = [org_stats[t].item() / stats[t].item() for t in top_tokens][1:]
print(ckpt_acc)
print(org_acc)
plt.plot(range(k-1), ckpt_acc, 'o', markersize=10, label="distill")
plt.plot(range(k-1), org_acc, 'x', markersize=10, label="org")
plt.legend()
plt.ylabel("Acuuracy")
plt.savefig("compare")
# print(len(sorted_pairs))

# model_path = "/data/vicuna-7b-v1.3/"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokens = tokenizer.convert_ids_to_tokens([x[0] for x in sorted_pairs[:k]])
# print("=========Top Tokens===========")
# print(tokens)
# print([x[0] for x in sorted_pairs[:k]])

# plt.bar(range(k), [x[1] for x in sorted_pairs[:k]])
# plt.ylabel("Frequency")
# plt.savefig("1")
# print(len(sorted_pairs))

# model_path = "/data/vicuna-7b-v1.3/"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokens = tokenizer.convert_ids_to_tokens([x[0] for x in sorted_pairs[:k]])
# print("=========Top Tokens===========")
# print(tokens)
# print([x[0] for x in sorted_pairs[:k]])
