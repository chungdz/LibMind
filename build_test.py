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
    for imp_id, hist, imp in tqdm(df[["id", "hist", "imp"]].values, total=df.shape[0]):
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
        clabel = 0
        for impre in imp_list:
            curn = news_info[impre]
            label = clabel
            clabel = clabel ^ 1

            new_row = []
            new_row.append(int(imp_id))
            new_row.append(label)
            new_row.append(curn)
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
        output_path = os.path.join("data", args.fout,  "{}-{}.npy".format(args.ftype, i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="test/behaviors.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--ftype", default="test", type=str,
                        help="train or dev")
    parser.add_argument("--max_hist_length", default=100, type=int,
                        help="Max length of the click history of the user.")
    parser.add_argument("--processes", default=40, type=int,
                        help="Processes number")

    args = parser.parse_args()

    main(args)

