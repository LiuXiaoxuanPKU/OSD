import matplotlib.pyplot as plt

c = 0.1


def get_speedup(gamma, alpha, c):
    return (1-alpha**(gamma+1))/(1-alpha)/(gamma*c+1)


fig, ax = plt.subplots()
fig.set_size_inches(3, 3)
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for gamma in [3, 5, 7, 9]:
    speedups = []
    for alpha in alphas:
        speedups.append(get_speedup(gamma, alpha, c))
    plt.plot(alphas, speedups, label=f"k={gamma} (c={c})")
plt.xticks(fontsize=12)
plt.xlabel("Alpha", size=14)
plt.yticks(fontsize=12)
plt.ylabel("Expected Speedup", size=14)
ax.axhline(y=1, color='r',
           linestyle='--', linewidth=2)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.savefig("graphs/analysis_k.pdf")
plt.close()


gamma = 5
fig, ax = plt.subplots()
fig.set_size_inches(3, 3)
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for c in [0.01, 0.05, 0.1, 0.2]:
    speedups = []
    for alpha in alphas:
        speedups.append(get_speedup(gamma, alpha, c))
    plt.plot(alphas, speedups, label=f"c={c}(k={gamma})")
plt.xticks(fontsize=12)
plt.xlabel("Alpha", size=14)
plt.yticks(fontsize=12)
plt.ylabel("Expected Speedup", size=14)
ax.axhline(y=1, color='r',
           linestyle='--', linewidth=2)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.savefig("graphs/analysis_c.pdf")
