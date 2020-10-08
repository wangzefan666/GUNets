import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_act(act):
    if act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky':
        return nn.LeakyReLU()
    elif act == 'none':
        return lambda x: x
    else:
        raise ValueError('Trans Act %s not defined' % act)


def init_layer(layer, init_function):
    if init_function != 'none':
        if init_function == 'xavier':
            nn.init.xavier_uniform_(layer.weight)
        elif init_function == 'kaiming':
            nn.init.kaiming_normal_(layer.weight)


class GUN(nn.Module):
    def __init__(self, num_feature, num_classes, emb_size=128, un_layer=2, bn_mom=0.1, drop_rate=0.5, device='cpu',
                 if_trans_bias=False, if_trans_bn=False, trans_init='xavier', trans_act='leaky',
                 if_trans_share=True, if_bn_share=True,
                 mlp_size=128, mlp_layer=1, if_mlp_bn=True, mlp_init='xavier', mlp_act='leaky'):
        super(GUN, self).__init__()

        self.emb_size = emb_size
        self.drop_out = torch.nn.Dropout(drop_rate).to(device)
        self.device = device
        self.bn = torch.nn.BatchNorm1d(num_feature, momentum=bn_mom).to(device)
        self.if_trans_share = if_trans_share
        self.if_trans_bn = if_trans_bn
        self.if_bn_share = if_bn_share

        # TRANS LINEAR
        if if_trans_share:
            self.trans_lin = nn.Linear(num_feature, emb_size, bias=if_trans_bias).to(device)
            init_layer(self.trans_lin, trans_init)
        else:
            self.trans_lin = nn.ModuleList()
            for i in range(un_layer + 1):
                linear = nn.Linear(num_feature, emb_size, bias=if_trans_bias).to(device)
                init_layer(linear, trans_init)
                self.trans_lin.append(linear)
        if if_trans_bn:
            if if_bn_share:
                self.trans_bn = torch.nn.BatchNorm1d(emb_size, momentum=bn_mom).to(device)
            else:
                self.trans_bn = torch.nn.BatchNorm1d((un_layer + 1) * emb_size, momentum=bn_mom).to(device)

        # Trans Activation
        self.trans_act = get_act(trans_act)

        # MLP Activation
        self.mlp_act = get_act(mlp_act)

        # MLP
        mlp_list = []
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = (un_layer + 1) * emb_size
            else:
                pre_size = mlp_size
            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            init_layer(linear, mlp_init)
            mlp_list.append(linear)
            if if_mlp_bn:
                mlp_list.append(nn.BatchNorm1d(mlp_size, momentum=bn_mom).to(device))
            mlp_list.extend([self.mlp_act, nn.Dropout(p=drop_rate)])

        if mlp_layer <= 1:
            pre_size = (un_layer + 1) * emb_size
        else:
            pre_size = mlp_size

        linear = torch.nn.Linear(pre_size, num_classes, bias=True).to(device)
        init_layer(linear, mlp_init)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)

    def forward(self, X):
        batch = X.shape[0]
        input_dim = X.shape[1]
        X = self.bn(X.reshape([batch * input_dim, -1])).contiguous().view([batch, input_dim, -1])
        X = self.drop_out(X)
        if self.if_trans_share:
            trans_x = self.trans_act(self.trans_lin(X))
        else:
            trans_list = []
            for i in range(input_dim):
                trans_list.append(self.trans_act(self.trans_lin[i](X[:, i, :])))
            trans_x = torch.cat(trans_list, dim=-1)
        if self.if_trans_bn:
            if self.if_bn_share:
                trans_x = self.trans_bn(trans_x.reshape([batch * input_dim, -1])).reshape([batch, -1])
            else:
                trans_x = self.trans_bn(trans_x.reshape([batch, -1]))

        trans_x = self.drop_out(trans_x)
        # concat the entries' features and put them input mlp2
        pred_x = self.mlp(trans_x)
        return pred_x
