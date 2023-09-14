from collectior import Collector
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/data/starcoderbase/")

def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    input_ids = tokenizer(case['content'])["input_ids"]
    if len(input_ids) < 200:
        return None
    prompt = tokenizer.decode(input_ids[:200])
    label = tokenizer.decode(input_ids[200:])
    if need_label:
        case["conversation"] = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    else:
        case["conversation"] = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    return case


if __name__ == "__main__":
    data_name = "bigcode/starcoderdata"
    language = "rust"
    c = Collector(data_name, data_dir=language)
    c.collect("train", transform, size=10000, prefix=language)