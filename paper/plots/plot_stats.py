import pickle as pk
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import seaborn as sns
import numpy as np


def main(dataset):
    with open(f"data/org-{dataset}.pkl", 'rb') as file:
        org_stats = pk.load(file)

    with open(f"data/teacher-fwd-{dataset}.pkl", 'rb') as file:
        distill_stats = pk.load(file)

    # assert torch.allclose(org_stats["generated"], distill_stats["generated"])
    
    # pick the most frequent 100 tokens
    k = 100
    topk = org_stats["generated"].topk(k)
    top_tokens = topk.indices[1:]
    cum_percentage =sum(topk.values) * 1.0 / sum(org_stats["generated"])
    print(cum_percentage)
    
    def precision_and_recall(stats, tokens):
        precision = [stats["correct"][t].item() / max(stats["proposed"][t].item(), 1) for t in tokens]
        recall = [stats["correct"][t].item() / max(stats["generated"][t].item(), 1) for t in tokens]
        return precision, recall
    
    org_tp_precision, org_tp_recall = precision_and_recall(org_stats, top_tokens)
    distill_tp_precision, distill_tp_recall = precision_and_recall(distill_stats, top_tokens)
    
    def plot_precision_recall():
        # precision
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 3)
        label_size = 14
        plt.plot(range(k-1), distill_tp_precision, '^', markersize=10, label="Distilled")
        plt.plot(range(k-1), org_tp_precision, 'x', markersize=10, label="Original")
        plt.legend(loc="upper center", ncol=2,
                   fontsize=12,
                bbox_to_anchor=(0.5,1.27))
        plt.ylabel("Precision", size=label_size)
        plt.xlabel("Top 100 frequent tokens", size=label_size)
        plt.tight_layout()
        plt.savefig("graphs/precision.pdf")
        plt.close()
        
        # recall
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 3)
        label_size = 14
        plt.plot(range(k-1), distill_tp_recall, '^', markersize=10, label="Distilled")
        plt.plot(range(k-1), org_tp_recall, 'x', markersize=10, label="Original")
        plt.legend(loc="upper center", ncol=2, 
                   fontsize=12,
                bbox_to_anchor=(0.5,1.27))
        plt.ylabel("Recall", size=label_size)
        plt.xlabel("Top 100 frequent tokens", size=label_size)
        plt.tight_layout()
        plt.savefig("graphs/recall.pdf")
        plt.close()

    plot_precision_recall()
    
    # model_path = "/rscratch/zhendong/lily/vicuna-7b-v1.3/"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # org_all_precision, org_all_recall = precision_and_recall(org_stats, range(tokenizer.vocab_size))
    # distill_all_precision, distill_all_recall = precision_and_recall(distill_stats, range(tokenizer.vocab_size))
    # precision_delta = torch.tensor([a - b for (a, b) in zip(distill_all_precision, org_all_precision)])
    # recall_delta = torch.tensor([a - b for (a, b) in zip(distill_all_recall, org_all_recall)])
    # precision_pairs = [t for t in precision_delta.topk(32000, sorted=True).indices.tolist() if t in top_tokens]
    # recall_pairs = [t for t in recall_delta.topk(32000, sorted=True).indices.tolist() if t in top_tokens]
    # # print(precision_pairs)
    # # exit(0)
    
    # print(tokenizer.convert_ids_to_tokens(tokenizer("SELECT AVG(students) from table").input_ids))
    # tokens = tokenizer.convert_ids_to_tokens(precision_pairs)
    # print("=========Precision Top Tokens===========")
    # print(tokens)
    
    # tokens = tokenizer.convert_ids_to_tokens(recall_pairs)
    # print("=========Recall Top Tokens===========")
    # print(tokens)

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


if __name__ == "__main__":
    dataset = "spider"
    main(dataset)