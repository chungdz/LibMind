# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020-10-02
training script

CUDA_VISIBLE_DEVICES=0,1,2 python training.py training.gpus=3
"""
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
from torchfm.model.wd import WideAndDeepModel
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM


def run(cfg, rank, device, train_dataset, valid_dataset):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    # Build Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # # Build model.
    if cfg.model == 'fm':
        fleid_dims = []
        for t in range(cfg.max_hist_length + 1):
            fleid_dims.append(cfg.news_num)
        model = FactorizationMachineModel(fleid_dims, 100)
        print('load FactorizationMachineModel')
        model.to(device)
    elif cfg.model == 'dfm':
        fleid_dims = []
        mlp_dims = []
        for t in range(cfg.max_hist_length + 1):
            fleid_dims.append(cfg.news_num)
            mlp_dims.append(100)
        model = DeepFactorizationMachineModel(fleid_dims, 100, mlp_dims, 0.2)
        print('load DeepFactorizationMachineModel')
        model.to(device)
    elif cfg.model == 'wd':
        fleid_dims = []
        mlp_dims = []
        for t in range(cfg.max_hist_length + 1):
            fleid_dims.append(cfg.news_num)
            mlp_dims.append(100)
        model = WideAndDeepModel(fleid_dims, 100, mlp_dims, 0.2)
        print('load WideAndDeepModel')
        model.to(device)
    elif cfg.model == 'ctr_dfm':
        fix_f = [SparseFeat('target_news', cfg.news_num, embedding_dim=100)]
        var_f = [VarLenSparseFeat(SparseFeat('his_news', vocabulary_size=cfg.news_num, embedding_dim=100), maxlen=cfg.max_hist_length, combiner='sum')]
        f = fix_f + var_f
        print('load ctr dfm')
        model = DeepFM(f, f, task='binary', device=device)
    elif cfg.model == 'ctr_fm':
        fix_f = [SparseFeat('target_news', cfg.news_num, embedding_dim=100)]
        var_f = [VarLenSparseFeat(SparseFeat('his_news', vocabulary_size=cfg.news_num, embedding_dim=100), maxlen=cfg.max_hist_length, combiner='sum')]
        f = fix_f + var_f
        print('load ctr fm')
        model = DeepFM(f, f, task='binary', device=device)
        model.use_dnn = False

    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    validate(cfg, -1, model, device, rank, valid_data_loader, fast_dev=True)
    for epoch in range(cfg.epoch):
        
        train(cfg, epoch, rank, model, train_data_loader, optimizer, steps_one_epoch, device)
        validate(cfg, epoch, model, device, rank, valid_data_loader)
        save_checkpoint_by_epoch(model.state_dict(), epoch, cfg.checkpoint_path)


def train(cfg, epoch, rank, model, loader, optimizer, steps_one_epoch, device):
    """
    train loop
    :param args: config
    :param epoch: int, the epoch number
    :param gpu_id: int, the gpu id
    :param rank: int, the process rank, equal to gpu_id in this code.
    :param model: gating_model.Model
    :param loader: train data loader.
    :param criterion: loss function
    :param optimizer:
    :param steps_one_epoch: the number of iterations in one epoch
    :return:
    """
    model.train()

    model.zero_grad()
    enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    for i, data in enum_dataloader:
        if i >= steps_one_epoch:
            break
        
        data = data.to(device)
        pred = model(data[:, 2:]).squeeze()
        loss = F.binary_cross_entropy(pred, data[:, 1].float())

        loss.backward()
        
        optimizer.step()
        
        model.zero_grad()



def validate(cfg, epoch, model, device, rank, valid_data_loader, fast_dev=False, top_k=20):
    model.eval()

    # Setting the tqdm progress bar
    if rank == 0:
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_test:%d" % epoch,
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")
    else:
        data_iter = enumerate(valid_data_loader)
                        
    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:
            if fast_dev and i > 10:
                break

            imp_ids += data[:, 0].cpu().numpy().tolist()
            data = data.to(device)

            # 1. Forward
            pred = model(data[:, 2:]).squeeze()

            preds += pred.cpu().numpy().tolist()
            truths += data[:, 1].long().cpu().numpy().tolist()

        print(len(preds))
        all_keys = list(set(imp_ids))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(truths, preds, imp_ids):
            group_labels[k].append(l)
            group_preds[k].append(p)
        
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        
        metric_list = [x.strip() for x in "group_auc || mean_mrr || ndcg@5;10".split("||")]
        ret = cal_metric(all_labels, all_preds, metric_list)
        for metric, val in ret.items():
            print("Epoch: {}, {}: {}".format(1, metric, val))

def main(cfg):
    
    set_seed(7)
    print('load dev dataset')
    dev_list = []
    for i in range(cfg.filenum):
        dev_list.append(np.load("data/raw/dev-{}.npy".format(i)))
    validate_dataset = FMData(np.concatenate(dev_list, axis=0))
    print('load train dataset')
    train_dataset = FMData(np.load("data/raw/train-0-new.npy"))
    print('load news dict')
    news_dict = json.load(open('./data/news.json', 'r', encoding='utf-8'))
    cfg.news_num = len(news_dict)
    cfg.result_path = './result/'
    cfg.checkpoint_path = './checkpoint/'
    
    run(cfg, 0, 'cuda:0', train_dataset=train_dataset, valid_dataset=validate_dataset)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenum', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--weight_decay', type=float, default=1e-6)  
    parser.add_argument("--max_hist_length", default=100, type=int, help="Max length of the click history of the user.")
    parser.add_argument("--model", default='fm', type=str)
    opt = parser.parse_args()
    logging.warning(opt)

    main(opt)
