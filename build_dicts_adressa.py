import os
import json
import pickle
import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

# Path options.
parser = argparse.ArgumentParser()
parser.add_argument("--root", default="Adressa", type=str)
parser.add_argument("--max_title", default=10, type=int)
args = parser.parse_args()

print("Loading news info")
news_dict_raw = json.load(open('Adressa/news_dict.json'))

news_dict = {'<pad>': 0}
word_dict = {'<pad>': 0}
word_idx = 1
news_idx = 1
for nid, ninfo in tqdm(news_dict_raw.items(), total=len(news_dict_raw), desc='parse news'):

    news_dict[nid] = {}
    news_dict[nid]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(ninfo).split()
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < args.max_title:
        for l in range(args.max_title - cur_len):
            wid_arr.append(0)
    news_dict[nid]['title'] = wid_arr[:args.max_title]   

print('all word', len(word_dict))
json.dump(news_dict, open('{}/news.json'.format(args.root), 'w', encoding='utf-8'))
json.dump(word_dict, open('{}/word.json'.format(args.root), 'w', encoding='utf-8'))

print("Loading behaviors info")
behaviors_raw = json.load(open('{}/his_behaviors.json'.format(args.root)))

user_dict = {}
user_idx = 0
for uid, uinfo in tqdm(behaviors_raw.items(), total=len(behaviors_raw), desc='history behavior'):
    user_dict[uid] = {"his": [], "idx": user_idx}
    user_idx += 1
    
    for nid in uinfo['pos']:
        user_dict[uid]["his"].append(nid)

json.dump(user_dict, open('{}/user.json'.format(args.root), 'w', encoding='utf-8'))
