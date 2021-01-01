# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LibFM(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, news_num, dim_num=100):
        super(LibFM, self).__init__()
        self.embeddingL = nn.Embedding(news_num, 1)
        self.embeddingQ = nn.Embedding(news_num, dim_num)
        self.bias = nn.parameter.Parameter(torch.randn((1)).type(torch.FloatTensor))
        self.news_num = news_num

    def forward(self, X):
        # X batch_size, 1 + his_len
        eL = self.embeddingL(X)
        logitL = eL.sum(dim=1, keepdim=True)

        eQ = self.embeddingQ(X)
        logitFM1 = eQ.mul(eQ).sum(1, keepdim=True).sum(2, keepdim=True)

        z = eQ.sum(1, keepdim=True)
        z2 = z.mul(z)
        logitFM2 = z2.sum(dim=2, keepdim=True)

        logitFM = (logitFM1 - logitFM2) * 0.5

        logit = (logitL + logitFM).squeeze()
        logit = logit + self.bias.expand(1, logit.size()[0]).view(-1)
  
        return torch.sigmoid(logit)
