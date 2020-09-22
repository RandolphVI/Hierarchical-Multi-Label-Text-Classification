# -*- coding:utf-8 -*-
__author__ = 'randolph'

"""HAM unit."""

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_units, att_dim, num_classes):
        super(Attention, self).__init__()
        # Attention
        self.fc1 = nn.Linear(input_units, att_dim, bias=False)
        self.fc2 = nn.Linear(att_dim, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, input_x):
        O_h = torch.tanh(self.fc1(input_x))
        attention_matrix = self.fc2(O_h).transpose(1, 2)
        attention_weight = torch.softmax(attention_matrix, dim=2)
        attention_out = torch.matmul(attention_weight, input_x)
        attention_out = torch.mean(attention_out, dim=1)
        return attention_weight, attention_out


class LocalLayer(nn.Module):
    def __init__(self, input_units, num_classes):
        super(LocalLayer, self).__init__()
        # Attention
        self.local = nn.Linear(input_units, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        self.local.weight.data.normal_(0, 0.1)

    def forward(self, input_x, input_att_weight):
        logits = self.local(input_x)
        scores = torch.sigmoid(logits)
        K = torch.mul(input_att_weight, scores.unsqueeze(-1))
        visual = torch.mean(K, dim=1)
        visual = torch.softmax(visual, dim=-1)
        return logits, scores, visual


class HAM(nn.Module):
    def __init__(self, input_units, att_dim, rnn_dim, fc_dim, num_classes):
        super(HAM, self).__init__()

        self.att = Attention(input_units, att_dim, num_classes)
        self.fc = nn.Sequential(nn.Linear(rnn_dim * 4, fc_dim), nn.ReLU())
        self.local = LocalLayer(fc_dim, num_classes)

    def forward(self, input_x, rnn_avg):
        att_weight, att_out = self.att(input_x)
        fc_in = torch.cat((rnn_avg, att_out), dim=1)
        fc_out = self.fc(fc_in)
        logits, scores, visual = self.local(fc_out, att_weight)
        return fc_out, scores, visual