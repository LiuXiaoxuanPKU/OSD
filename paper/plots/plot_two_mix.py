import argparse
import seaborn as sns
import math
import wandb
import matplotlib.pyplot as plt

run_name = "lily-falcon/spec/xewjxcma"

def moving_average(data, window_size):
    return [sum(data[i:i+window_size])/min(len(data)-i, window_size) for i in range(len(data))]

def plot_run():
    api = wandb.Api()
    run = api.run(run_name)
    data = []
    alpha_gsm8k = []
    alpha_finance = []

    i = 0
    for row in run.scan_history():
        if "train/global_step" not in row:
            continue
        if "alpha_gsm8k" in row and row["alpha_gsm8k"] is not None:
            alpha_gsm8k.append((row["train/global_step"], row["alpha_gsm8k"]))
        if "alpha_finance" in row and row["alpha_finance"] is not None:
            alpha_finance.append((row["train/global_step"], row["alpha_finance"]))
    window_size = 50
    gsm8k_x = [x[0] for x in alpha_gsm8k][:len(alpha_gsm8k)][::20]
    # print(max(gsm8k_x))
    alpha_gsm8k = moving_average([d[1] for d in alpha_gsm8k], window_size)[::20]
    # print(len(gsm8k_x), len(alpha_gsm8k))
    finance_x = [x[0] for x in alpha_finance][::20]
    alpha_finance = moving_average([d[1] for d in alpha_finance], window_size)[::20]
    # print(len(finance_x), len(alpha_finance))
    
    ######################## Start Plot ############################
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3)
    tick_size = 14
    label_size = 14
    max_x = int(max(gsm8k_x) // 1000) + 2
    plt.xticks([1000 * x for x in range(max_x)], 
               list(range(max_x)),
               fontsize=tick_size)
    # sns.scatterplot(x=range(len(alphas)), y=alphas)
    ax.set_ylim(0.55, 0.8)
    plt.yticks(fontsize=tick_size)
    sns.lineplot(x=gsm8k_x, y=alpha_gsm8k)
    sns.lineplot(x=finance_x, y=alpha_finance)
    plt.ylabel("Alpha", size=label_size)
    plt.xlabel("# of Records (K)", size=label_size)
    plt.axvspan(0, 2000, color='#EAAA60', alpha=0.4, label='Update with\nGsm8K')
    plt.axvspan(2000, 4000, color='#7DA6C6', alpha=0.4, label='Update with\nAlpaca-finance')
    ax.text( 1000 , 0.76, 
            "Update with\nGsm8K", 
            horizontalalignment='center', 
            verticalalignment='center', 
            # transform=ax.transAxes,
            fontsize=label_size - 1)
    ax.text( 3000 , 0.76, 
            'Update with\nAlpaca-finance', 
            horizontalalignment='center', 
            verticalalignment='center', 
            # transform=ax.transAxes,
            fontsize=label_size - 1)
    plt.tight_layout()
        
    ax.text(0.98, 0.03, 
            "Mix of two", 
            horizontalalignment='right', 
            verticalalignment='bottom', 
            transform=ax.transAxes,
            fontsize=label_size + 2)
    plt.tight_layout()
    plt.savefig(f"graphs/mix.pdf")
      
if __name__ == "__main__":
    plot_run()