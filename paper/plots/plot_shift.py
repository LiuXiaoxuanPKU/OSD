import argparse
import seaborn as sns
import math
import wandb
import matplotlib.pyplot as plt

DATASET_TO_RUN = {
    # "smooth" : "lily-falcon/specInfer/93r0g2ti",
    "sharp": "lily-falcon/spec/52hkohc7",
    # "sharp-10": "lily-falcon/spec/1w2wfao0",
    "sharp-30": "lily-falcon/spec/po19ram0",
    "sharp-50": "lily-falcon/spec/tdud2367",
    "sharp-70": "lily-falcon/spec/q3hhz4uu",
    "sharp-100": "lily-falcon/spec/c11961i8",
}


DATASET_TO_NAME = {
    # "smooth" : "Smooth Shift",
    "sharp": "Online",
    # "sharp-10" : "10%",
    "sharp-30": "30%",
    "sharp-50": "50%",
    "sharp-70": "70%",
    "sharp-100": "100%",
}


def moving_average(data, window_size):
    return [sum(data[max(0, i - window_size):i]) / min(window_size, i) for i in range(1, len(data) + 1)]


def plot_run(run_name, dataset, linewidth):
    api = wandb.Api()
    run = api.run(run_name)
    data = []
    for row in run.scan_history(keys=["alpha", "_step", "train/global_step"]):
        if math.isnan(row["alpha"]):
            continue
        data.append((row["train/global_step"], row["alpha"]))
    # filter out data
    alphas = moving_average([d[1] for d in data], 100)

    ######################## Start Plot ############################
    if linewidth == 5:
        linestyle = '-'
        alpha = 1
    else:
        linestyle = '-'
        alpha = 0.6
    sns.lineplot(x=[x for x in range(len(alphas))],
                 y=alphas, label=DATASET_TO_NAME[dataset],
                 linestyle=linestyle, linewidth=linewidth,
                 alpha=alpha)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)
    tick_size = 14
    label_size = 14
    max_x = 9
    plt.xticks([1000 * x for x in range(max_x)],
               list(range(max_x)),
               fontsize=tick_size)
    ax.set_ylim(0.45, 0.8)
    plt.yticks(fontsize=tick_size)

    for dataset in DATASET_TO_RUN:
        if dataset == "sharp":
            linewidth = 5
        else:
            linewidth = 2
        plot_run(DATASET_TO_RUN[dataset], dataset, linewidth)

    plt.legend(ncol=2, fontsize=label_size-1)
    label_size = 14
    plt.ylabel("Alpha", size=label_size)
    plt.xlabel("# of Records (K)", size=label_size)
    # plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.axvspan(0, 2000, color='#EAAA60', alpha=0.4)
    plt.axvspan(2000, 4000, color='#E68B81', alpha=0.4)
    plt.axvspan(4000, 6000, color='#B7B2D0', alpha=0.4)
    plt.axvspan(6000, 8000, color='#7DA6C6', alpha=0.4)
    datasets = ["Gsm8k", "Spider", "Alpaca-\nfinance", "Code-search-\npython"]
    for i in range(4):
        ax.text(i*2000+1000, 0.822,
                datasets[i],
                horizontalalignment='center',
                verticalalignment='center',
                weight='bold',
                fontsize=label_size - 1)

    plt.tight_layout()
    plt.savefig(f"graphs/sharp.pdf")
