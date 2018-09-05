# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class TD_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TD_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.lstm_entity1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim*2, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_entity2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim*2, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim*2, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(opt.hidden_dim*6, opt.hidden_dim)
        # self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        x_e1, x_e2, x_c = inputs[0], inputs[1], inputs[2]
        x_e1_len, x_e2_len, x_c_len = torch.sum(x_e1 != 0, dim=-1), torch.sum(x_e2 != 0, dim=-1), torch.sum(x_c != 0, dim=-1)
        x_e1, x_e2, x_c = self.embed(x_e1), self.embed(x_e2), self.embed(x_c)
        _, (h_n_e1, _) = self.lstm_entity1(x_e1, x_e1_len)
        _, (h_n_e2, _) = self.lstm_entity2(x_e2, x_e2_len)
        _, (h_n_c, _) = self.lstm_context(x_c, x_c_len)
        h_n = torch.cat((h_n_e1[0], h_n_e2[0], h_n_c[0]), dim=-1)
        # h_n = torch.cat((h_n_e, h_n_c[0]), dim=-1)
        out = self.dense(h_n)
        # out = self.softmax(out)
        return out
