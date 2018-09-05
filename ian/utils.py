import os
import ast
import spacy
import numpy as np
from errno import ENOENT
from collections import Counter

from gensim.models import KeyedVectors

nlp = spacy.load("en")


def get_data_info(train_fname, test_fname, save_fname, pre_processed):
    word2id, max_entities_len, max_context_len = {}, 0, 0
    word2id['<pad>'] = 0
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 4:
                    max_entities1_len = int(content[1])
                    max_entities2_len = int(content[2])
                    max_context_len = int(content[3])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        lines = open(train_fname, 'r').readlines()
        for i in range(0, len(lines), 4):
            sptoks = nlp(lines[i].strip())
            words.extend([sp.text.lower() for sp in sptoks])
            if len(sptoks) - 1 > max_context_len:
                max_context_len = len(sptoks) - 1
            sptoks = nlp(lines[i + 1].strip())
            if len(sptoks) > max_entities_len:
                max_entities1_len = len(sptoks)
            sptoks = nlp(lines[i + 2].strip())
            if len(sptoks) > max_entities_len:
                max_entities_len = len(sptoks)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word and '\n' not in word and 'entity1_term' not in word and 'entity2_term' not in word:
                word2id[word] = len(word2id)
    
        lines = open(test_fname, 'r').readlines()
        for i in range(0, len(lines), 4):
            sptoks = nlp(lines[i].strip())
            words.extend([sp.text.lower() for sp in sptoks])
            if len(sptoks) - 1 > max_context_len:
                max_context_len = len(sptoks) - 1
            sptoks = nlp(lines[i + 1].strip())
            if len(sptoks) > max_entities_len:
                max_entities_len = len(sptoks)
            sptoks = nlp(lines[i + 1].strip())
            if len(sptoks) > max_entities_len:
                max_entities_len = len(sptoks)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word and '\n' not in word and 'entity_term' not in word and 'entity2_term' not in word:
                word2id[word] = len(word2id)

        with open(save_fname, 'w') as f:
            f.write('length %s %s\n' % (max_entities_len, max_context_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))
                
    print('There are %s words in the dataset, the max length of entities is %s, and the max length of context is %s' % (len(word2id), max_entities_len, max_context_len))
    return word2id, max_entities_len, max_context_len

def read_data(fname, word2id, max_entities_len, max_context_len, save_fname, pre_processed):
    entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens = list(), list(), list(), list(), list(), list(), list()
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 7):
            entities1.append(ast.literal_eval(lines[i]))
            entities2.append(ast.literal_eval(lines[i + 1]))
            contexts.append(ast.literal_eval(lines[i + 2]))
            labels.append(ast.literal_eval(lines[i + 3]))
            entities1_lens.append(ast.literal_eval(lines[i + 4]))
            entities2_lens.append(ast.literal_eval(lines[i + 5]))
            context_lens.append(ast.literal_eval(lines[i + 6]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        lines = open(fname, 'r').readlines()
        with open(save_fname, 'w') as f:
            for i in range(0, len(lines), 4):
                polarity = lines[i + 3].split()[0]
                if polarity == 'conflict':
                    continue

                context_sptoks = nlp(lines[i].strip())
                context = []
                for sptok in context_sptoks:
                    if sptok.text.lower() in word2id:
                        context.append(word2id[sptok.text.lower()])

                entities1_sptoks = nlp(lines[i + 1].strip())
                entity1 = []
                for entity_sptok in entities1_sptoks:
                    if entity_sptok.text.lower() in word2id:
                        entity1.append(word2id[entity_sptok.text.lower()])

                entities2_sptoks = nlp(lines[i + 2].strip())
                entity2 = []
                for entity_sptok in entities2_sptoks:
                    if entity_sptok.text.lower() in word2id:
                        entity2.append(word2id[entity_sptok.text.lower()])

                entities1.append(entity1 + [0] * (max_entities_len - len(entity1)))
                f.write("%s\n" % entities1[-1])
                entities2.append(entity2 + [0] * (max_entities_len - len(entity2)))
                f.write("%s\n" % entities2[-1])
                contexts.append(context + [0] * (max_context_len - len(context)))
                f.write("%s\n" % contexts[-1])
                # if polarity == 'negative':
                #     labels.append([1, 0, 0])
                # elif polarity == 'neutral':
                #     labels.append([0, 1, 0])
                # elif polarity == 'positive':
                #     labels.append([0, 0, 1])
                if str(polarity.strip()) == '1':
                    labels.append([1, 0])
                elif str(polarity.strip()) == '0':
                    labels.append([0, 1])
                f.write("%s\n" % labels[-1])
                entities1_lens.append(len(entities1_sptoks))
                f.write("%s\n" % entities1_lens[-1])
                entities2_lens.append(len(entities2_sptoks))
                f.write("%s\n" % entities2_lens[-1])
                context_lens.append(len(context_sptoks) - 1)
                f.write("%s\n" % context_lens[-1])

    print("Read %s examples from %s" % (len(entities1), fname))
    print("entities 1 shape = " + str(np.asarray(entities1).shape))
    print("entities 2 shape = " + str(np.asarray(entities2).shape))
    print("context shape = " + str(np.asarray(contexts).shape))
    print("labels shape = " + str(np.asarray(labels).shape))
    print("entities 1 len shape = " + str(np.asarray(entities1_lens).shape))
    print("entities 2 len shape = " + str(np.asarray(entities2_lens).shape))
    print("context len shape = " + str(np.asarray(context_lens).shape))
    return np.asarray(entities1), np.asarray(entities2), np.asarray(contexts), np.asarray(labels), np.asarray(entities1_lens), np.asarray(entities2_lens), np.asarray(context_lens)

def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'rb') as f:
        for line in f:
            try:
                line = line.decode('utf-8')
                content = line.strip().split()
                if content[0] in word2id:
                    word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                    oov = oov - 1
            except:
                print("Exception occured!")
                continue
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

def load_word_embeddings_w2v(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)
    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)

    w2v_model = KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in w2v_model.vocab:
        if word in word2id:
            word2vec[word2id[word]] = w2v_model[word]
            oov = oov - 1

    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]