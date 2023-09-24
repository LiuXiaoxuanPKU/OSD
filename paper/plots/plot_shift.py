import argparse
import seaborn as sns
import math
import wandb
import matplotlib.pyplot as plt

DATASET_TO_RUN = {
    # "smooth" : "lily-falcon/specInfer/93r0g2ti",
    "sharp" : "lily-falcon/spec/52hkohc7",
}


DATASET_TO_NAME = {
    # "smooth" : "Smooth Shift",
    "sharp" : "Sharp Shift",
}

def moving_average(data, window_size):
    return [sum(data[i:i+window_size])/min(len(data)-i, window_size) for i in range(len(data))]

def plot_run(run_name, dataset):
    api = wandb.Api()
    run = api.run(run_name)
    data = []
    for row in run.scan_history(keys=["alpha", "_step"]):
        if math.isnan(row["alpha"]):
            continue
        data.append((row["_step"], row["alpha"]))
    # filter out data
    alphas = moving_average([d[1] for d in data], 50)
    
    
    ######################## Start Plot ############################
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3)
    tick_size = 14
    label_size = 14
    max_x = int(len(alphas) // 1000) + 3
    plt.xticks([1000 * x for x in range(max_x)], 
               list(range(max_x)),
               fontsize=tick_size)
    # sns.scatterplot(x=range(len(alphas)), y=alphas)
    ax.set_ylim(0.4, 0.8)
    plt.yticks(fontsize=tick_size)
    sns.lineplot(x=[x for x in range(len(alphas))], y=alphas)
    plt.ylabel("Alpha", size=label_size)
    plt.xlabel("# of Records (K)", size=label_size)
    plt.tight_layout()
    # plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.axvspan(0, 2000, color='#EAAA60', alpha=0.4, label='Gsm8k')
    plt.axvspan(2000, 4000, color='#E68B81', alpha=0.4, label='Spider')
    plt.axvspan(4000, 6000, color='#B7B2D0', alpha=0.4, label='Alpaca-finance')
    plt.axvspan(6000, 8000, color='#7DA6C6', alpha=0.4, label='Code-search-python')
    # plt.legend(loc='upper center', 
    #            bbox_to_anchor=(0.5, 1.2), 
    #            ncol=4, fontsize=10)
    datasets = ["Gsm8k", "Spider", "Alpaca-\nfinance", "Code-search-\npython"]
    for i in range(4):
        ax.text( i*2000+1000 , 0.75, 
            datasets[i], 
            horizontalalignment='center', 
            verticalalignment='center', 
            # transform=ax.transAxes,
            fontsize=label_size - 1)
        
    ax.text(0.98, 0.03, 
            DATASET_TO_NAME[dataset], 
            horizontalalignment='right', 
            verticalalignment='bottom', 
            transform=ax.transAxes,
            fontsize=label_size + 2)
    plt.tight_layout()
    plt.savefig(f"graphs/{dataset}.pdf")
      
if __name__ == "__main__":
    # plot(args.fielname)
    for dataset in DATASET_TO_RUN:
        plot_run(DATASET_TO_RUN[dataset], dataset)