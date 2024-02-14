import os, sys
import json
import sqlite3
import traceback
import argparse

import random
import numpy as np

import re

import sqlite3

file_path_1 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_1.json"
file_path_2 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_2.json"
file_path_3 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_3.json"
file_path_4 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_4.json"
file_path_5 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_5.json"
file_path_5 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_6.json"
file_path_5 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_7.json"
file_path_5 = "vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_8.json"

def merge_two_json(file_path_1, file_path_2, file_path_3, file_path_4, file_path_5, file_path_6, file_path_7, file_path_8):

    f_read_1 = json.load(open(file_path_1, "r"))
    f_read_2 = json.load(open(file_path_2, "r"))
    f_read_3 = json.load(open(file_path_3, "r"))
    f_read_4 = json.load(open(file_path_4, "r"))
    f_read_5 = json.load(open(file_path_5, "r"))
    f_read_5 = json.load(open(file_path_6, "r"))
    f_read_5 = json.load(open(file_path_7, "r"))
    f_read_5 = json.load(open(file_path_8, "r"))

    print(len(f_read_1))
    print(len(f_read_2))
    print(len(f_read_3))
    print(len(f_read_4))
    print(len(f_read_5))
    print(len(f_read_6))
    print(len(f_read_7))
    print(len(f_read_8))

    f_all = f_read_1 + f_read_2 + f_read_3 + f_read_4 + f_read_5 + f_read_6 + f_read_7 + f_read_8
    random.shuffle(f_all)
    print(len(f_all))

    with open('vicuna160m_chatbot_arena_all_token_acceptance_rate_prepare_for_training_264k.json', 'w') as f_merged:
        json.dump(f_all, f_merged)

if __name__ == "__main__":
    merge_two_json(file_path_1, file_path_2, file_path_3, file_path_4, file_path_5, file_path_6, file_path_7, file_path_8)