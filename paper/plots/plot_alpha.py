import math
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_TO_RUN = {
    "spider" : "lily-falcon/spec/0gzouckc",
    "gsm8k" : "lily-falcon/spec/t1y2v4hu",
    "python" : "lily-falcon/spec/bemti3sf",
    "finance" : "lily-falcon/spec/5ldgffvr",
}

BASELINES = {
    "spider" : "lily-falcon/specInfer/gwgk68hk",
    "gsm8k" : "lily-falcon/spec/mld0iqfk",
    "python" : "lily-falcon/spec/7cbgoxpe",
    "finance" : "lily-falcon/spec/440jir4a",
}

OFFLINE = {
    "spider" : 0.76,
    "gsm8k": 0.75,
    "python":0.65,
    "finance": 0.67
}

LOW_BOUNDS = {
   "spider" : 0.3,
    "gsm8k": 0.5,
    "python":0.3,
    "finance": 0.5 
}


# # 100% accuracy
# FULL_APLHA = {
#    "spider" : TODO,
#    "gsm8k" : TODO,
#     "python" : TODO,
#     "finance" : TODO
# }


DATASET_TO_NAME = {
    "spider" : "Spider",
    "gsm8k" : "Gsm8k",
    "python" : "Code-search-python",
    "finance" : "Alpaca-finance"
}

def load_run(run):
    def moving_average(data, window_size):
        return [sum(data[max(0, i - window_size):i]) / min(window_size, i) for i in range(1, len(data) + 1)]
    
    steps = []
    alphas = []
    i = 0
    for row in run.scan_history(keys=["alpha", "_step"]):
        i += 1
        if math.isnan(row["alpha"]):
            continue
        steps.append(row["_step"]//2)
        alphas.append(row["alpha"])
    # filter out data
    alphas = moving_average(alphas, 50)
    steps = steps[::30]
    alphas = alphas[::30]
    return steps, alphas

def plot_run(dataset):
    api = wandb.Api()
    run = api.run(DATASET_TO_RUN[dataset])
    steps, alphas = load_run(run)
    baseline_run = api.run(BASELINES[dataset])
    baseline_steps, baseline_alphas = load_run(baseline_run)
    
    ######################## Start Plot ############################
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 3)
    tick_size = 14
    label_size = 14
    max_x = int(min(max(steps), max(baseline_steps)) // 1000) + 1
    # print(max_x, max(baseline_steps), min(baseline_steps))
    plt.xticks(
               [1000 * x for x in range(0, max_x, 2)], 
               list(range(0, max_x, 2)),
               fontsize=tick_size)
    ax.set_xlim(0, max_x*1000)
    ax.set_ylim(LOW_BOUNDS[dataset], 0.8)
    plt.yticks(fontsize=tick_size)
    # plt.plot(steps, alphas, "o-", markersize=8)
    sns.lineplot(x=steps, y=alphas)
    sns.lineplot(x=baseline_steps, y=baseline_alphas)
    ax.axhline(y=OFFLINE[dataset], color='r', 
               linestyle='--', linewidth=2)

    plt.ylabel("Alpha", size=label_size)
    plt.xlabel("# of Records (K)", size=label_size)
    plt.tight_layout()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.text(0.98, 0.03, 
            DATASET_TO_NAME[dataset], 
            horizontalalignment='right', 
            verticalalignment='bottom', 
            transform=ax.transAxes,
            fontsize=label_size + 2)
    plt.savefig(f"graphs/{dataset}.pdf")
      
if __name__ == "__main__":
    # plot(args.fielname)
    for dataset in DATASET_TO_RUN:
        # if dataset not in ["spider", "gsm8k", "python"]:
        #     continue
        plot_run(dataset)