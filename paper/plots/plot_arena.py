import math
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_TO_RUN = {
    "chinese" : "lily-falcon/spec/cdwvh8nu",
    "japanese" : "lily-falcon/spec/nlsl4spm",
    "spanish" : "lily-falcon/spec/gk0j7cye",
    "portuguese" : "lily-falcon/spec/j8yg9nx4",
    "russian": "lily-falcon/spec/9rkala4p"
}


# # 100% accuracy
# FULL_APLHA = {
#    "spider" : TODO,
#    "gsm8k" : TODO,
#     "python" : TODO,
#     "finance" : TODO
# }

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
    alphas = moving_average(alphas, 40)
    steps = steps[::30]
    alphas = alphas[::30]
    return steps, alphas

def plot_runs():
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 3)
    tick_size = 14
    label_size = 14
    
    for dataset in DATASET_TO_RUN:
        api = wandb.Api()
        run = api.run(DATASET_TO_RUN[dataset])
        steps, alphas = load_run(run)
        sns.lineplot(x=steps, y=alphas, label=dataset.capitalize())
        
    ######################## Start Plot ############################
    max_x = 2
    plt.xticks(
               [500 * x for x in range(0, max_x * 2)], 
               [x/2.0 for x in range(0, max_x * 2)],
               fontsize=tick_size)
    ax.set_xlim(0, max_x*1000)
    ax.set_ylim(0.3, 0.75)
    plt.yticks(fontsize=tick_size)

    plt.legend(ncol=3, 
               loc="upper center", 
               bbox_to_anchor=(0.5,1.14),
               fontsize=10)
    plt.ylabel("Alpha", size=label_size)
    plt.xlabel("# of Records (K)", size=label_size)
    plt.tight_layout()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.text(0.98, 0.03, 
            "Chatbot-Arena", 
            horizontalalignment='right', 
            verticalalignment='bottom', 
            transform=ax.transAxes,
            fontsize=label_size + 2)
    plt.savefig(f"graphs/arena.pdf")
      
if __name__ == "__main__":
    plot_runs()