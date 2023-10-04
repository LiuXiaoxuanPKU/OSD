from transformers import pipeline
import json

filename = "clean_chat_clean_conv_20230809_100k.json"
with open(filename, "r") as f:
    data = json.load(f)

classes = {}
count = 0
classifier = pipeline("text-classification", model="alimazhar-110/website_classification")
for i, d in enumerate(data):
    try:
        if d['language'] != 'English':
            continue
        prompt = d['conversation'][0]['content']
        out = classifier(prompt)
        label = out[0]['label']
        if label not in classes:
            classes[label] = []
        classes[label].append(d)
    except:
        count += 1
        print(f"ignore {count}/{i}")

for c in classes:
    print(c, len(classes[c]))
    filename = c.replace("/", "_")
    with open(filename+".json", "w") as f:
        json.dump(f"raw_data/{classes[c]}", f)
