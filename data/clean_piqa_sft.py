from collector import Collector


def transform(i, case, need_label=False):
    case["id"] = f"identity_{i}"
    if need_label:
        case["text"] = case['goal'] + case[f"sol{1 + int(case['label'])}"]
    else:
        case["text"] = case['goal']
    return case


if __name__ == "__main__":
    data_name = "piqa"
    c = Collector(data_name)
    c.collect("train", transform)
    c.collect("test", transform, 200)