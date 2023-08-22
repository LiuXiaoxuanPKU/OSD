import json
import random

filename = "clean_chat_clean_conv_20230809_10k.json"
with open(filename, "r") as f:
    data = json.load(f)

random.shuffle(data)
split_idx = int(0.98 * len(data))

train_data = data[:split_idx]
eval_data = data[split_idx:]

# sort data based on timestamp
sorted_train_data = sorted(train_data, key=lambda x: x['tstamp'])
sorted_eval_data = sorted(eval_data, key=lambda x: x['tstamp'])

with open('train.json', 'w') as f:
    json.dump(sorted_train_data, f)
    
with open('eval.json', 'w') as f:
    json.dump(sorted_eval_data, f)