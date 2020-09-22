# -*- coding:utf-8 -*-
__author__ = 'randolph'

"""HmcNet layers."""

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

sys.path.append('../')

from torch.autograd import Variable
from torch.nn.utils import weight_norm
from ham import HAM


class BiRNNLayer(nn.Module):
    def __init__(self, input_units, rnn_type, rnn_layers, rnn_hidden_size, dropout_keep_prob):
        super(BiRNNLayer, self).__init__()
        if rnn_type == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                  batch_first=True, bidirectional=False, dropout=dropout_keep_prob)
        if rnn_type == 'GRU':
            self.bi_rnn = nn.GRU(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

    def forward(self, input_x, input_xlens):
        """
        RNN Layer.
        Args:
            input_x: [batch_size, batch_max_seq_len, embedding_size]
            input_xlens: The ground truth lengths of each sequence
        Returns:
            rnn_out: [batch_size, batch_max_seq_len, rnn_hidden_size]
            rnn_avg: [batch_size, rnn_hidden_size]
        """
        rnn_out, _ = self.bi_rnn(input_x)
        temp = []
        batch_size = input_x.size()[0]
        for i in range(batch_size):
            word_state = rnn_out[i, :input_xlens[i], :]
            avg_state = torch.mean(word_state, dim=0)
            temp.append(avg_state)
        rnn_avg = torch.stack(temp, 0)
        return rnn_out, rnn_avg


class HighwayLayer(nn.Module):
    def __init__(self, in_units, out_units):
        super(HighwayLayer, self).__init__()
        self.fc_h = nn.Linear(in_features=in_units, out_features=out_units, bias=True)
        self.fc_t = nn.Linear(in_features=in_units, out_features=out_units, bias=True)
        self.init_weights()

    def init_weights(self):
        self.fc_t.bias.data.fill_(-2.0)

    def forward(self, input_x):
        highway_h = torch.relu(self.fc_h(input_x))
        highway_t = torch.sigmoid(self.fc_t(input_x))
        highway_out = torch.mul(highway_h, highway_t) + torch.mul((1 - highway_t), input_x)
        return highway_out


class Loss(nn.Module):
    # TODO
    def __init__(self):
        super(Loss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='mean')
        self.BCELoss = nn.BCELoss(reduction='mean')

    def forward(self, local_predict_y, global_predict_y, local_y, global_y):
        # Loss
        local_losses = 0.0
        for index, predict_y in enumerate(local_predict_y):
            local_losses += self.BCELoss(predict_y, local_y[index])

        global_losses = self.BCELoss(global_predict_y, global_y)
        losses = local_losses + global_losses
        return losses


class HmcNet(nn.Module):
    """An implementation of HyperNet"""
    def __init__(self, args, vocab_size, embedding_size, pretrained_embedding=None):
        super(HmcNet, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self._setup_layers()

    def _setup_embedding_layer(self):
        """
        Creating Embedding layers.
        """
        if self.pretrained_embedding is None:
            embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(self.vocab_size, self.args.embedding_dim)))
            embedding_weight = Variable(embedding_weight, requires_grad=True)
        else:
            if self.args.embedding_type == 0:
                embedding_weight = torch.from_numpy(self.pretrained_embedding).float()
            if self.args.embedding_type == 1:
                embedding_weight = Variable(torch.from_numpy(self.pretrained_embedding).float(), requires_grad=True)
        self.embedding = nn.Embedding(self.vocab_size, self.args.embedding_dim, _weight=embedding_weight)

    def _setup_bi_rnn_layer(self):
        """
        Creating Bi-RNN Layer.
        """
        self.bi_rnn = BiRNNLayer(input_units=self.args.embedding_dim, rnn_type=self.args.rnn_type,
                                 rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                 dropout_keep_prob=self.args.dropout_rate)

    def _setup_ham_unit(self):
        """
        Creating HAM unit.
        """
        # TODO
        self.first_ham = HAM(input_units=self.args.rnn_dim * 2, att_dim=self.args.attention_dim,
                             rnn_dim=self.args.rnn_dim, fc_dim=self.args.fc_dim,
                             num_classes=self.args.num_classes_list[0])
        self.second_ham = HAM(input_units=self.args.rnn_dim * 2, att_dim=self.args.attention_dim,
                              rnn_dim=self.args.rnn_dim, fc_dim=self.args.fc_dim,
                              num_classes=self.args.num_classes_list[1])
        self.third_ham = HAM(input_units=self.args.rnn_dim * 2, att_dim=self.args.attention_dim,
                             rnn_dim=self.args.rnn_dim, fc_dim=self.args.fc_dim,
                             num_classes=self.args.num_classes_list[2])
        self.fourth_ham = HAM(input_units=self.args.rnn_dim * 2, att_dim=self.args.attention_dim,
                              rnn_dim=self.args.rnn_dim, fc_dim=self.args.fc_dim,
                              num_classes=self.args.num_classes_list[3])

    def _setup_highway_layer(self):
        """
         Creating Highway Layer.
         """
        self.highway = HighwayLayer(in_units=self.args.fc_dim, out_units=self.args.fc_dim)

    def _setup_fc_layer(self):
        """
         Creating FC Layer.
         """
        self.fc = nn.Sequential(nn.Linear(in_features=self.args.fc_dim * 4,
                                          out_features=self.args.fc_dim, bias=True),
                                nn.ReLU())
        self.out = nn.Linear(in_features=self.args.fc_dim, out_features=self.args.total_classes, bias=True)

    def _setup_dropout(self):
        """
         Adding Dropout.
         """
        self.dropout = nn.Dropout(self.args.dropout_rate)

    def _setup_cal_loss(self):
        """
        Calculate Loss.
        """
        self.loss = Loss()

    def _setup_layers(self):
        """
        Creating layers of model.
        1. Embedding Layer.
        2. Bi-RNN Layer.
        3. Hierarchical Attention-based Recurrent Layer
        4. Highway Layer.
        5. FC Layer.
        6. Dropout
        """
        self._setup_embedding_layer()
        self._setup_bi_rnn_layer()
        self._setup_ham_unit()
        self._setup_highway_layer()
        self._setup_fc_layer()
        self._setup_dropout()
        self._setup_cal_loss()

    def forward(self, record):
        x, xlens, sec, subsec, group, subgroup, y = record
        local_y = sec, subsec, group, subgroup

        _pad, _len = rnn_utils.pad_packed_sequence(x, batch_first=True)

        embedded_sentence = self.embedding(_pad)
        # shape of embedded_sentence: [batch_size, batch_max_len, embedding_size]
        embedded_sentence = embedded_sentence.view(embedded_sentence.shape[0], embedded_sentence.shape[1], -1)

        # Bi-RNN Layer
        # shape of rnn_out: [batch_size, rnn_hidden_size]
        rnn_out, rnn_avg = self.bi_rnn(embedded_sentence, xlens)

        # HAR Layer
        # First Level
        first_out, first_scores, first_visual = self.first_ham(rnn_out, rnn_avg)

        # Second Level
        second_ham_input = torch.mul(rnn_out, first_visual.unsqueeze(-1))
        second_out, second_scores, second_visual = self.second_ham(second_ham_input, rnn_avg)

        # Third Level
        third_ham_input = torch.mul(rnn_out, second_visual.unsqueeze(-1))
        third_out, third_scores, third_visual = self.third_ham(third_ham_input, rnn_avg)

        # Fourth Level
        fourth_ham_input = torch.mul(rnn_out, third_visual.unsqueeze(-1))
        fourth_out, fourth_scores, fourth_visual = self.fourth_ham(fourth_ham_input, rnn_avg)

        local_scores_list = first_scores, second_scores, third_scores, fourth_scores

        # Concat
        # shape of ham_out: [batch_size, fc_dim * 4]
        ham_out = torch.cat((first_out, second_out, third_out, fourth_out), dim=1)

        # Fully Connected Layer
        fc_out = self.fc(ham_out)

        # Highway Layer
        highway_out = self.highway(fc_out)

        # Dropout
        h_drop = self.dropout(highway_out)

        global_logits = self.out(h_drop)
        global_scores = torch.sigmoid(global_logits)

        losses = self.loss(local_scores_list, global_scores, local_y, y)

        return global_logits, global_scores, losses



