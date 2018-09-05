# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class IAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.lstm_entity1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_entity2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.attention_entity1 = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_entity2 = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context1 = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context2 = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, entities1_indices, entities2_indices = inputs[0], inputs[1], inputs[2]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        entities1_len = torch.sum(entities1_indices != 0, dim=-1)
        entities2_len = torch.sum(entities2_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        entities1 = self.embed(entities1_indices)
        entities2 = self.embed(entities2_indices)
        entities1, (_, _) = self.lstm_entity1(entities1, entities1_len)
        entities2, (_, _) = self.lstm_entity2(entities2, entities2_len)
        context, (_, _) = self.lstm_context(context, text_raw_len)

        entities1_len = torch.tensor(entities1_len, dtype=torch.float).to(self.opt.device)
        entities1 = torch.sum(entities1, dim=1)
        entities1 = torch.div(entities1, entities1_len.view(entities1_len.size(0), 1))

        entities2_len = torch.tensor(entities2_len, dtype=torch.float).to(self.opt.device)
        entities2 = torch.sum(entities2, dim=1)
        entities2 = torch.div(entities2, entities2_len.view(entities2_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context = torch.sum(context, dim=1)
        context = torch.div(context, text_raw_len.view(text_raw_len.size(0), 1))

        # entities1_final = self.attention_entity1(entities1, entities2).squeeze(dim=1)
        # entities2_final = self.attention_entity2(entities2, entities1).squeeze(dim=1)
        # entities_final = entities1_final * entities2_final
        # context_final = self.attention_context(context, entities_final).squeeze(dim=1)
        # entities_final = torch.mm(entities1, entities2.transpose(1, 0))
        # context_final = self.attention_context(context, entities1_final).squeeze(dim=1)

        context_final1 = self.attention_context1(context, entities1).squeeze(dim=1)
        context_final2 = self.attention_context2(context, entities2).squeeze(dim=1)

        x = torch.cat((context_final1, context_final2), dim=-1)
        # x = torch.cat((x_e, context_final), dim=-1)
        out = self.dense(x)
        return out
