import os, sys
import json
import sqlite3
import traceback
import argparse

import random
import numpy as np

import re

import sqlite3

file_path = "ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"

def reconfigure_json(file_path):

    f_read = json.load(open(file_path, "r"))

    print(len(f_read))
    
    f_all = []

    i = 0
    for d in f_read:
        print(len(d['conversations']))
        if len(d['conversations']) <= 1:
            continue
        
        d['conversation'] = d.pop('conversations')
        
        #print(d['conversation'])
        
        for i, source in enumerate(d['conversation']):
            #print(source)
            source['content'] = source.pop('value')

        #print(d['conversation'])
        
        f_all.append(d)
        i += 1

    with open('raw_data/sharegpt_all.json', 'w') as f_merged:
        json.dump(f_all, f_merged, indent=4)

if __name__ == "__main__":
    reconfigure_json(file_path)
