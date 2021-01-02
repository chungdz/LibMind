import os
import argparse
import json
import pickle
from tqdm import tqdm
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
import math
from dataset import FMData
from gather import gather as gather_all
from libfm import LibFM
from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.dfm import DeepFactorizationMachineModel

def run(cfg, rank, test_dataset, device, model):
    set_seed(7)

    model.to(device)
    model.eval()

    # test_dataset = DistValidationDataset(cfg.dataset, start, end, "test_set_dist", "test_set-{}.pt".format(check_point_offset))
    valid_data_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False)

    if ((cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0)):
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_test:%d" % 1,
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")
    else:
        data_iter = enumerate(valid_data_loader)

    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:


            imp_ids += data[:, 0].cpu().numpy().tolist()
            data = data.to(device)

            # 1. Forward
            pred = model(data[:, 2:])

            preds += pred.cpu().numpy().tolist()
            truths += data[:, 1].long().cpu().numpy().tolist()

        tmp_dict = {}
        tmp_dict['imp'] = imp_ids
        tmp_dict['labels'] = truths
        tmp_dict['preds'] = preds

        with open(cfg.result_path + 'tmp_small_{}.json'.format(rank), 'w', encoding='utf-8') as f:
            json.dump(tmp_dict, f)

def gather(cfg, turn, validate=False):
    output_path = cfg.result_path
    filenum = cfg.gpus

    preds = []
    labels = []
    imp_indexes = []

    for i in range(filenum):
        with open(output_path + 'tmp_small_{}.json'.format(i), 'r', encoding='utf-8') as f:
            cur_result = json.load(f)
        imp_indexes += cur_result['imp']
        labels += cur_result['labels']
        preds += cur_result['preds']

    tmp_dict = {}
    tmp_dict['imp'] = imp_indexes
    tmp_dict['labels'] = labels
    tmp_dict['preds'] = preds

    with open(cfg.result_path + 'tmp_{}.json'.format(turn), 'w', encoding='utf-8') as f:
        json.dump(tmp_dict, f)


def split_dataset(dataset, gpu_count):
    sub_len = math.ceil(len(dataset) / gpu_count)
    data_list = []
    for i in range(gpu_count):
        s = i * sub_len
        e = (i + 1) * sub_len
        data_list.append(dataset[s: e])

    return data_list


def main(cfg):
    set_seed(7)

    file_num = cfg.filenum
    cfg.result_path = './result/'
    print('load dict')
    news_dict = json.load(open('./data/news.json', 'r', encoding='utf-8'))
    cfg.news_num = len(news_dict)

    if cfg.model = 'fm':
        fleid_dims = []
        for t in range(cfg.max_hist_length + 1):
            fleid_dims.append(cfg.news_num)
        model = FactorizationMachineModel(fleid_dims, 100)
        model.to(device)
    elif cfg.model = 'dfm':
        fleid_dims = []
        mlp_dims = []
        for t in range(cfg.max_hist_length + 1):
            fleid_dims.append(cfg.news_num)
            mlp_dims.append(100)
        model = DeepFactorizationMachineModel(fleid_dims, 100, mlp_dims, 0.2)
        model.to(device)

    saved_model_path = os.path.join('./checkpoint/', 'model.ep{0}'.format(cfg.epoch))
    print("Load from:", saved_model_path)
    if not os.path.exists(saved_model_path):
        print("Not Exist: {}".format(saved_model_path))
        return []
    model.cpu()
    pretrained_model = torch.load(saved_model_path, map_location='cpu')
    print(model.load_state_dict(pretrained_model, strict=False))

    for point_num in range(file_num):
        print("processing data/raw/test-{}.npy".format(point_num))
        valid_dataset = FMData(np.load("data/raw/test-{}.npy".format(point_num)))

        dataset_list = split_dataset(valid_dataset, cfg.gpus)
        
        processes = []
        for rank in range(cfg.gpus):
            cur_device = torch.device("cuda:{}".format(rank))

            p = mp.Process(target=run, args=(cfg, rank, dataset_list[rank], cur_device, model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        gather(cfg, point_num)
    
    gather_all(cfg.result_path, file_num, validate=False, save=True)
        



if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenum', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--gpus', type=int, default=2, help='gpu_num')
    parser.add_argument('--epoch', type=int, default=0, help='the number of epochs load checkpoint')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--port', type=int, default=9337)
    parser.add_argument("--max_hist_length", default=100, type=int, help="Max length of the click history of the user.")
    parser.add_argument("--model", default='fm', type=str)
    opt = parser.parse_args()
    logging.warning(opt)

    main(opt)
