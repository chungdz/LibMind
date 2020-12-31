import os
import json
import pickle
import argparse

import pandas as pd
import numpy as np

f_train_beh = os.path.join("data", "train/behaviors.tsv")
all_beh = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

json.load(open('data/news.json', 'r', encoding='utf-8'))
json.load(open('data/user.json', 'r', encoding='utf-8'))
