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
all_news = pd.read_csv("{}/news.tsv".format(args.root), sep="\t", encoding="utf-8",
                            names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                            quoting=3)

news_dict = {'<pad>': 0}
word_dict = {'<pad>': 0}
word_idx = 1
news_idx = 1
for n, title in all_news[['newsid', "title"]].values:
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(title).split()
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
    news_dict[n]['title'] = wid_arr[:args.max_title]

print('all word', len(word_dict))
json.dump(news_dict, open('{}/news.json'.format(args.root), 'w', encoding='utf-8'))
json.dump(word_dict, open('{}/word.json'.format(args.root), 'w', encoding='utf-8'))

f_train_beh = "{}/train_behaviors.tsv".format(args.root)
f_dev_beh = "{}/dev_behaviors.tsv".format(args.root)
f_test_beh = "{}/test_behaviors.tsv".format(args.root)

print("Loading training beh")
all_beh = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
print("Loading dev beh")
dev_beh = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
all_beh = pd.concat([all_beh, dev_beh], ignore_index=True)
print("Loading testing beh")
test_beh = pd.read_csv(f_test_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
all_beh = pd.concat([all_beh, test_beh], ignore_index=True)

user_dict = {}
user_idx = 0
for uid, hist in tqdm(all_beh[["uid", "hist"]].values, total=all_beh.shape[0], desc='build user'):
    user_dict[uid] = {
        'idx': user_idx,
        'his': hist.split()
    }
    user_idx += 1

print("user num", user_idx)

json.dump(user_dict, open('{}/user.json'.format(args.root), 'w', encoding='utf-8'))
