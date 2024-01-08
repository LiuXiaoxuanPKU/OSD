from datasets import load_dataset
import json

def transform_dataset(data_name):
    dataset = load_dataset(data_name, split='train')
    #print(dataset)
    all_data = []
    for d in dataset:
        #print(d)
        conv_a = d.pop('conversation_a')
        model_a = d.pop('model_a')
        win_flag_a = True if d['winner'] == 'model_a' else False


        conv_b = d.pop('conversation_b')
        model_b = d.pop('model_b')
        win_flag_b = True if d['winner'] == 'model_b' else False

        d.pop('winner')

        #print(conv_a)
        #print(conv_b)

        new_d_1 = d.copy()
        new_d_2 = d.copy()

        new_d_1['conversation'] = conv_a
        new_d_2['conversation'] = conv_b

        new_d_1['model'] = model_a
        new_d_2['model'] = model_b

        new_d_1['foe_model'] = model_b
        new_d_2['foe_model'] = model_a

        new_d_1['winner'] = win_flag_a
        new_d_2['winner'] = win_flag_b

        all_data.append(new_d_1)
        all_data.append(new_d_2)

    with open('raw_data/chatbot_arena_token_acceptance_rate_testing.json', 'w') as f_merged:
        json.dump(all_data, f_merged)

if __name__ == "__main__":
    data_name = "lmsys/chatbot_arena_conversations"
    transform_dataset(data_name)
