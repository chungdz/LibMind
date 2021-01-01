import os
import json
import pickle
import argparse
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm

random.seed(7)

def build_examples(rank, args, df, news_info, fout):
    data_list = []
    for hist, imp in tqdm(df[["hist", "imp"]].values, total=df.shape[0]):
        if str(hist) == 'nan':
            his_list = []
        else:
            his_list = str(hist).strip().split()
            
        his_idx_list = [news_info[h] for h in his_list]
        hislen = len(his_idx_list)
        if hislen < args.max_hist_length:
            for _ in range(args.max_hist_length - hislen):
                his_idx_list.append(0)
        else:
            his_idx_list = his_idx_list[-args.max_hist_length:]

        imp_list = str(imp).split(' ')
        imp_pos_list = []
        imp_neg_list = []
        for impre in imp_list:
            arr = impre.split('-')
            curn = news_info[arr[0]]
            label = int(arr[1])
            if label == 0:
                imp_neg_list.append((curn, label))
            elif label == 1:
                imp_pos_list.append((curn, label))
            else:
                raise Exception('label error!')
        # down sample
        neg_num = math.ceil(len(imp_neg_list) / 5)
        sampled = random.sample(imp_neg_list, neg_num)
        all_imp = imp_pos_list + sampled
        for p in all_imp:
            new_row = []
            new_row.append(p[0])
            new_row.append(p[1])
            new_row += his_idx_list
            data_list.append(new_row)
    
    datanp = np.array(data_list, dtype=int)
    np.save(fout, datanp)

def main(args):
    f_train_beh = os.path.join("data", args.fsamples)
    df = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    news_info = json.load(open('data/news.json', 'r', encoding='utf-8'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join("data", args.fout, "training_set_dist-{}.npy".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="train/behaviors.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--max_hist_length", default=100, type=int,
                        help="Max length of the click history of the user.")
    parser.add_argument("--processes", default=40, type=int,
                        help="Processes number")

    args = parser.parse_args()

    main(args)

