import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

filename = "full_bench.pk"
results = pk.load(open(filename, "rb"))
prompt_times = [results[l][0] * 1.0 for l in results]
decode_times = [np.median(results[l][1:]) for l in results]
prompt_lens = [l/1000 for l in list(results.keys())]
prompt_percentage = [results[l][0] / sum(results[l]) for l in results]
# print(results)
plt.figure(figsize=(4, 4))
# plt.scatter(prompt_lens, prompt_times, label=f"prompt phase", s=10)
# plt.scatter(prompt_lens, decode_times, label=f"decode phase", s=10)
plt.scatter(prompt_lens, prompt_percentage)
for l, p in zip(prompt_lens, decode_times):
    print(l, p)
plt.xlabel("Prompt length (K)")
plt.ylabel("Time (s/token)")
plt.ylabel("Prompt Phase Percentage(%)")
# plt.legend()
plt.tight_layout()
plt.savefig("full_time")