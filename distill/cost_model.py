def search_best_spec_count(acc_rate, speed_ratio):
    best_speedup = 0
    best_spec_count = 0
    for i in range(1, 100):
        speedup = estimate_speedup_from_acc_rate(acc_rate, i, speed_ratio)
        if speedup > best_speedup:
            best_speedup, best_spec_count = speedup, i
    return int(best_spec_count), best_speedup
    
def acc_rate_to_acc_count(acc_rate, spec_count):
    return (1 - acc_rate**(spec_count + 1)) / (1 - acc_rate)

def estimate_speedup_from_acc_count(acc_count, spec_count, speed_ratio):
    return acc_count / (speed_ratio * spec_count + 1)

def estimate_speedup_from_acc_rate(acc_rate, spec_count, speed_ratio):
    acc_count = acc_rate_to_acc_count(acc_rate, spec_count)
    return estimate_speedup_from_acc_count(acc_count, spec_count, speed_ratio)


if __name__ == "__main__":
    speed_ratios = [0.01, 0.02, 0.05, 0.1]
    acc_rate = 0.6
    best_spec_counts = [7, 6, 4, 3]
    for i, sr in enumerate(speed_ratios):
        assert search_best_spec_count(acc_rate, sr)[0] == best_spec_counts[i], \
            f"{search_best_spec_count(acc_rate, sr)[0]} != {best_spec_counts[i]}"
    

