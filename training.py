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
from deepctr_torch.models import DeepFM, WDL

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run(cfg, rank, device, finished, train_dataset_path, valid_dataset):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    print("Worker %d is setting dataset ... " % rank)
    # Build Dataloader
    train_dataset = FMData(np.load(train_dataset_path))
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # MIND 用ID adressa 不用
    # # Build model.
    fix_f = [SparseFeat('target_news', cfg.news_num, embedding_dim=100)]
    var_f = [VarLenSparseFeat(SparseFeat('his_news', vocabulary_size=cfg.news_num, embedding_dim=100), maxlen=cfg.max_hist_length, combiner='sum')]
    f = fix_f + var_f
    # f = [] 
    f.append(VarLenSparseFeat(SparseFeat('target_title', vocabulary_size=cfg.word_num, embedding_dim=100), maxlen=cfg.max_title, combiner='sum'))
    f.append(VarLenSparseFeat(SparseFeat('his_title', vocabulary_size=cfg.word_num, embedding_dim=100), maxlen=cfg.max_hist_length * cfg.max_title, combiner='mean'))
    if cfg.model == 'ctr_dfm':
        print('load ctr dfm')
        model = DeepFM(f, f, task='binary', device=device)
    elif cfg.model == 'ctr_fm':
        print('load ctr fm')
        model = LibFM(f, f, task='binary', device=device)
    elif cfg.model == 'ctr_wdl':
        print('load ctr wdl')
        model = WDL(f, f, task='binary', device=device)

    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print("Worker %d is working ... " % rank)
    # Fast check the validation process
    if (cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0):
        validate(cfg, -1, model, device, rank, valid_data_loader, fast_dev=True)
        logging.warning(model)
        gather_all(cfg.result_path, 1, validate=True, save=False)
    
    # Training and validation
    for epoch in range(cfg.epoch):
        # print(model.match_prediction_layer.state_dict()['2.bias'])
        train(cfg, epoch, rank, model, train_data_loader,
              optimizer, steps_one_epoch, device)
    
        validate(cfg, epoch, model, device, rank, valid_data_loader)
        # add finished count
        finished.value += 1

        if (cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0):
            save_checkpoint_by_epoch(model.state_dict(), epoch, cfg.checkpoint_path)

            while finished.value < cfg.gpus:
                time.sleep(1)
            gather_all(cfg.result_path, cfg.gpus, validate=True, save=False)
            finished.value = 0

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


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

    enum_dataloader = enumerate(loader)
    if ((cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0)):
        enum_dataloader = enumerate(tqdm(loader, total=len(loader), desc="EP-{} train".format(epoch)))

    for i, data in enum_dataloader:
        if i >= steps_one_epoch:
            break
        # data = {key: value.to(device) for key, value in data.items()}
        data = data.to(device)
        # 1. Forward
        pred = model(data[:, 2:]).squeeze()
        loss = F.binary_cross_entropy(pred, data[:, 1].float())

        # 3.Backward.
        loss.backward()

        if cfg.gpus > 1:
            average_gradients(model)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        model.zero_grad()

    # if (not args.dist_train) or (args.dist_train and rank == 0):
    #     util.save_checkpoint_by_epoch(
    #         model.state_dict(), epoch, args.checkpoint_path)


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

        tmp_dict = {}
        tmp_dict['imp'] = imp_ids
        tmp_dict['labels'] = truths
        tmp_dict['preds'] = preds

        with open(cfg.result_path + 'tmp_{}.json'.format(rank), 'w', encoding='utf-8') as f:
            json.dump(tmp_dict, f)
        f.close()


def init_processes(cfg, local_rank, vocab, dataset, valid_dataset, finished, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    addr = "localhost"
    port = cfg.port
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=0 + local_rank,
                            world_size=cfg.gpus)

    device = torch.device("cuda:{}".format(local_rank))

    fn(cfg, local_rank, device, finished, train_dataset_path=dataset, valid_dataset=valid_dataset)


def split_dataset(dataset, gpu_count):
    sub_len = len(dataset) // gpu_count
    if len(dataset) != sub_len * gpu_count:
        len_a, len_b = sub_len * gpu_count, len(dataset) - sub_len * gpu_count
        dataset, _ = torch.utils.data.random_split(dataset, [len_a, len_b])

    return torch.utils.data.random_split(dataset, [sub_len, ] * gpu_count)

def split_valid_dataset(dataset, gpu_count):
    sub_len = math.ceil(len(dataset) / gpu_count)
    data_list = []
    for i in range(gpu_count):
        s = i * sub_len
        e = (i + 1) * sub_len
        data_list.append(dataset[s: e])

    return data_list

def main(cfg):
    
    set_seed(7)
    # print('load train')
    # train_list = []
    # for i in range(cfg.filenum):
    #     train_list.append(np.load("data/raw/train-{}.npy".format(i)))
    # train_dataset = FMData(np.concatenate(train_list, axis=0))
    print('load dev')
    dev_list = []
    for i in range(cfg.filenum):
        dev_list.append(np.load("{}/raw/{}-{}.npy".format(cfg.root, cfg.vtype, i)))
    validate_dataset = FMData(np.concatenate(dev_list, axis=0))
    print('load news dict')
    news_dict = json.load(open('./{}/news.json'.format(cfg.root, i), 'r', encoding='utf-8'))
    print('load words dict')
    word_dict = json.load(open('./{}/word.json'.format(cfg.root, i), 'r', encoding='utf-8'))
    cfg.news_num = len(news_dict)
    cfg.word_num = len(word_dict)
    cfg.result_path = './result/'
    cfg.checkpoint_path = './checkpoint/'
    finished = mp.Value('i', 0)

    assert(cfg.gpus > 1)
    # dataset_list = split_dataset(train_dataset, cfg.gpus)
    valid_dataset_list = split_valid_dataset(validate_dataset, cfg.gpus)

    processes = []
    for rank in range(cfg.gpus):
        p = mp.Process(target=init_processes, args=(
            cfg, rank, None, "{}/raw/train-{}-new.npy".format(cfg.root, rank), valid_dataset_list[rank], finished, run, "nccl"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenum', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--gpus', type=int, default=2, help='gpu_num')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--port', type=int, default=9337)
    parser.add_argument("--max_hist_length", default=100, type=int, help="Max length of the click history of the user.")
    parser.add_argument("--model", default='fm', type=str)
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--vtype", default="dev", type=str)
    parser.add_argument("--max_title", default=10, type=int)
    opt = parser.parse_args()
    logging.warning(opt)

    main(opt)
