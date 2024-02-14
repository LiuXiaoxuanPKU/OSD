import os, sys
import json
import sqlite3
import traceback
import argparse

import random
import numpy as np

import re

import sqlite3

file_path = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_8.json"

def reconfigure_json(file_path):

    f_read = json.load(open(file_path, "r"))

    print(len(f_read))

    f_all = f_read

    i = 0
    for d in f_all:
        records = d['sd_records']

        new_records = []
        for record_i in records:
            correct_cnt = record_i['accepted_len']

            record_i['predictor_labels'] = []
            for k in range(len(record_i['confidences'])):
                if k < correct_cnt:
                    record_i['predictor_labels'].append(1)
                else:
                    record_i['predictor_labels'].append(0)
            
            new_records.append(record_i)
        
        d['sd_records'] = new_records

        i += 1
        if i >= 500:
            break

    with open('/home/hedgehog/workspace/OSD/data/vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_eval_500_1.json', 'w') as f_merged:
        json.dump(f_all, f_merged, indent=4)

if __name__ == "__main__":
    reconfigure_json(file_path)