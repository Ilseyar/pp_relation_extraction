import argparse
from imp import reload

import tensorflow as tf
import numpy as np
from ian.utils import get_data_info, read_data, load_word_embeddings, load_word_embeddings_w2v
from ian.model import IAN

#corpora = ['twitter_adr', 'cadec', 'made', 'twimed/twitter', 'twimed/pubmed']
corpora = ['IEPA']
num_folds = 5


FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
# tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
# tf.app.flags.DEFINE_integer('n_epoch', 1, 'number of epoch')
# tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
# tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# tf.app.flags.DEFINE_integer('pre_processed', 1, 'Whether the data is pre-processed')
# tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
# tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
# tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')
#
# tf.app.flags.DEFINE_string('embedding_fname', '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/glove.840B.300d.txt', 'embedding file name')
# tf.app.flags.DEFINE_string('embedding_fname', '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/PubMed-w2v.bin', 'embedding file name')
# tf.app.flags.DEFINE_string('train_fname', 'data/' + corp + '/' + str(i + 1) + '/train.txt', 'training file name')
# tf.app.flags.DEFINE_string('test_fname', 'data/' + corp + '/' + str(i + 1) + '/test.txt', 'testing file name')
# tf.app.flags.DEFINE_string('data_info', 'data/' + corp + '/' + str(i + 1) + '/data_info.txt', 'the file saving data information')
# tf.app.flags.DEFINE_string('train_data', 'data/' + corp + '/' + str(i + 1) + '/train_data.txt', 'the file saving training data')
# tf.app.flags.DEFINE_string('test_data', 'data/' + corp + '/' + str(i + 1) + '/test_data.txt', 'the file saving testing data')


def run_classification():
    print('Loading data info ...')
    FLAGS.word2id, FLAGS.max_entities_len, FLAGS.max_context_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname, FLAGS.data_info, FLAGS.pre_processed)

    print('Loading training data and testing data ...')
    train_data = read_data(FLAGS.train_fname, FLAGS.word2id, FLAGS.max_entities_len, FLAGS.max_context_len, FLAGS.train_data, FLAGS.pre_processed)
    test_data = read_data(FLAGS.test_fname, FLAGS.word2id, FLAGS.max_entities_len, FLAGS.max_context_len, FLAGS.test_data, FLAGS.pre_processed)

    print('Loading pre-trained word vectors ...')
    FLAGS.word2vec = load_word_embeddings_w2v(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)

    #saver = tf.train.Saver()

    with tf.Session() as sess:
        model = IAN(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)

        #saver.save(sess, "/model/model.ckpt")

if __name__ == '__main__':
    for corp in corpora:
        print(corp)
        # out = open("output/result.txt", "a")
        # out.write("-------------" + corp + "-----------------" + "\n\n")
        # out.close()
        for i in range(0, num_folds):
            print(i)
            out = open("../output/result.txt", "a")
            out.write("fold " + str(i) + "\n")
            out.close()
            # tf.app.flags.DEFINE_string('train_fname', 'data/' + corp + '/folds/' + str(i + 1) + '/train.txt', 'training file name')
            # tf.app.flags.DEFINE_string('test_fname', 'data/' + corp + '/folds/' + str(i + 1) + '/test.txt', 'testing file name')
            # tf.app.flags.DEFINE_string('data_info', 'data/' + corp + '/folds/' + str(i + 1) + '/data_info.txt', 'the file saving data information')
            # tf.app.flags.DEFINE_string('train_data', 'data/' + corp + '/folds/' + str(i + 1) + '/train_data.txt', 'the file saving training data')
            # tf.app.flags.DEFINE_string('test_data', 'data/' + corp + '/folds/' + str(i + 1) + '/test_data.txt', 'the file saving testing data')

            tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
            tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
            tf.app.flags.DEFINE_integer('n_epoch', 15, 'number of epoch')
            tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
            tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
            tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
            tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
            tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
            tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

            # tf.app.flags.DEFINE_string('embedding_fname', '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/glove.840B.300d.txt', 'embedding file name')
            tf.app.flags.DEFINE_string('embedding_fname', '/media/ilseyar/Disk_D/Ilseyar/Projects/vectors/wikipedia-pubmed-and-PMC-w2v.bin', 'embedding file name')
            tf.app.flags.DEFINE_string('train_fname', '../data/' + corp + '/train_small.txt', 'training file name')
            tf.app.flags.DEFINE_string('test_fname', '../data/' + corp + '/test_small.txt', 'testing file name')
            tf.app.flags.DEFINE_string('data_info', '../data/' + corp + '/data_info_small.txt', 'the file saving data information')
            tf.app.flags.DEFINE_string('train_data', '../data/' + corp + '/train_data_small.txt', 'the file saving training data')
            tf.app.flags.DEFINE_string('test_data', '../data/' + corp + '/test_data_small.txt', 'the file saving testing data')
            #tf.app.run()
            run_classification()
            tf.app.flags.FLAGS = tf.flags._FlagValues()
            tf.app.flags._global_parser = argparse.ArgumentParser()
            FLAGS = tf.app.flags.FLAGS
            tf.reset_default_graph()

        # out1 = open("output/result_num.txt", "a")
        # out1.write("\n")
        # out1.close()