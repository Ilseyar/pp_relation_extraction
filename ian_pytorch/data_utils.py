# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        try:
            tokens = line.rstrip().split()
            if word2idx is None or tokens[0] in word2idx.keys():
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        except:
            print("Exception while load vectors")
            continue
    return word_vec

def load_word2vec(path, word2idx = None):
    word_vec = {}
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
    in_vocb = 0
    for word in w2v_model.vocab:
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = w2v_model[word]
            in_vocb += 1
    print("In vocabulary = %s, all words = %s", in_vocb, len(word2idx.keys()))
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
        embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    # if os.path.exists(embedding_matrix_file_name):
    #     print('loading embedding_matrix:', embedding_matrix_file_name)
    #     embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    # else:
        print('loading word vectors...')
        # embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        embedding_matrix = np.random.uniform(-0.01, 0.01, [len(word2idx) + 2, embed_dim])
        # fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
        # fname = '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/glove.840B.300d.txt'
            # if embed_dim != 300 else './glove.42B.300d.txt'
        # fname = '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/Health.s200.w10.n5.v10.cbow.bin'
        fname = '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/PubMed-w2v.bin'
        # word_vec = load_word_vec(fname, word2idx=word2idx)
        word_vec = load_word2vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            try:
                vec = word_vec.get(word)
                if vec is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = vec
            except:
                print("Exception while load vectors")
                continue
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
        return embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc, truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 4):
                text_left, _, text_middle = [s.lower().strip() for s in lines[i].partition("$T1$")]
                text_middle, _, text_right = [s.lower().strip() for s in text_middle.partition("$T2$")]
                entity1 = lines[i + 1].lower().strip()
                entity2 = lines[i + 2].lower().strip()
                text_raw = text_left + " " + entity1 + " " + text_middle + " " + entity2 + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 4):
            text_left, _, text_middle  = [s.lower().strip() for s in lines[i].partition("$T1$")]
            text_middle, _, text_right = [s.lower().strip() for s in text_middle.partition("$T2$")]
            entity1 = lines[i + 1].lower().strip()
            entity2 = lines[i + 2].lower().strip()
            polarity = lines[i + 3].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + "entity1" + " " + text_middle + " " + "entity2" + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_middle + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + entity1)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + entity2 + " " + text_right, reverse=True)
            text_middle_indices = tokenizer.text_to_sequence(text_middle)
            text_middle_with_aspect_indices = tokenizer.text_to_sequence(" " + entity1 + " " + text_middle + " " + entity2 + " ")
            entity1_indices = tokenizer.text_to_sequence(entity1)
            entity2_indices = tokenizer.text_to_sequence(entity2)
            # polarity = int(polarity)+1
            polarity = int(polarity)

            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_middle_indices': text_middle_indices,
                'text_middle_with_aspect_indices': text_middle_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'entity1_indices': entity1_indices,
                'entity2_indices': entity2_indices,
                'polarity': polarity,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=100, max_seq_len=40, fold_num = 1):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'aimed' : {
                'train': '../data/AiMed/folds/' + str(fold_num) + '/train_pytorch_short_context.txt',
                'test': '../data/AiMed/folds/' + str(fold_num) + '/test_pytorch_short_context.txt'
            },
            'cdr' :  {
                'train': '../data/CDR/train_short1.txt',
                'test': '../data/CDR/test_short1.txt'
            },
            'bioinfer' :  {
                'train': '../data/BioInfer/folds/' + str(fold_num) + '/train_pytorch.txt',
                'test': '../data/BioInfer/folds/' + str(fold_num) + '/test_pytorch.txt'
            }
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text.lower())
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))


