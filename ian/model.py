import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.python.ops import math_ops
import numpy as np
import time
from ian.utils import get_batch_index
from sklearn import metrics

class IAN(object):

    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout
        
        self.word2id = config.word2id
        self.max_entities_len = config.max_entities_len
        self.max_context_len = config.max_context_len
        self.word2vec = config.word2vec
        self.sess = sess

    def build_model(self):
        with tf.name_scope('inputs'):
            print(self.max_entities_len)
            print(self.max_context_len)
            self.entities1 = tf.placeholder(tf.int32, [None, self.max_entities_len])
            self.entities2 = tf.placeholder(tf.int32, [None, self.max_entities_len])
            self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.entities1_lens = tf.placeholder(tf.int32, None)
            self.entities2_lens = tf.placeholder(tf.int32, None)
            self.context_lens = tf.placeholder(tf.int32, None)
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            
            entities1_inputs = tf.nn.embedding_lookup(self.word2vec, self.entities1)
            entities1_inputs = tf.cast(entities1_inputs, tf.float32)
            entities1_inputs = tf.nn.dropout(entities1_inputs, keep_prob=self.dropout_keep_prob)

            entities2_inputs = tf.nn.embedding_lookup(self.word2vec, self.entities2)
            entities2_inputs = tf.cast(entities2_inputs, tf.float32)
            entities2_inputs = tf.nn.dropout(entities2_inputs, keep_prob=self.dropout_keep_prob)
            
            context_inputs = tf.nn.embedding_lookup(self.word2vec, self.contexts)
            context_inputs = tf.cast(context_inputs, tf.float32)
            context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)
        
        with tf.name_scope('weights'):
            weights = {
                'entities1_score': tf.get_variable(
                    name='W_e1',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'entities2_score' : tf.get_variable(
                    name='W_e2',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden * 2, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }
        
        with tf.name_scope('biases'):
            biases = {
                'entities1_score': tf.get_variable(
                    name='B_e1',
                    shape=[self.max_entities_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'entities2_score': tf.get_variable(
                    name='B_e2',
                    shape=[self.max_entities_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }
        
        with tf.name_scope('dynamic_rnn'):
            entities1_outputs, entities1_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=entities1_inputs,
                sequence_length=self.entities1_lens,
                dtype=tf.float32,
                scope='entities1_lstm'
            )
            batch_size = tf.shape(entities1_outputs)[0]
            entities1_avg = tf.reduce_mean(entities1_outputs, 1)

            entities2_outputs, entities2_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=entities2_inputs,
                sequence_length=self.entities2_lens,
                dtype=tf.float32,
                scope='entities2_lstm'
            )
            batch_size = tf.shape(entities2_outputs)[0]
            entities2_avg = tf.reduce_mean(entities2_outputs, 1)
            
            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=context_inputs,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            context_avg = tf.reduce_mean(context_outputs, 1)

            entities1_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            entities1_outputs_iter = entities1_outputs_iter.unstack(entities1_outputs)
            entities2_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            entities2_outputs_iter = entities2_outputs_iter.unstack(entities2_outputs)
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(context_avg)
            entities1_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            entities1_lens_iter = entities1_lens_iter.unstack(self.entities1_lens)
            entities2_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            entities2_lens_iter = entities2_lens_iter.unstack(self.entities2_lens)
            entities_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            entities_att = tf.TensorArray(size=batch_size, dtype=tf.float32)
            def body(i, entities_rep, entities_att):
                a1 = entities1_outputs_iter.read(i)
                a2 = entities2_outputs_iter.read(i)
                b = context_avg_iter.read(i)
                l1 = math_ops.to_int32(entities1_lens_iter.read(i))
                l2 = math_ops.to_int32(entities2_lens_iter.read(i))
                print(l1)
                e1 = tf.matmul(a1, weights['entities1_score'])
                e2 = tf.matmul(a2, weights['entities2_score'])
                e12 = tf.matmul(e1, e2)
                # entities_score = tf.reshape(tf.nn.tanh(tf.matmul(e12, tf.reshape(b, [-1, 1])) + biases['entities1_score'] + biases['entities2_score']), [1, -1])
                entities_score = tf.reshape(tf.nn.tanh(tf.matmul(e12, tf.reshape(b, [-1, 1])) + biases['context_score']), [1, -1])
                print(entities_score.shape)
                entities_att_temp = tf.concat([tf.nn.softmax(tf.slice(entities_score, [0, 0], [1, l1])), tf.zeros([1, self.max_entities_len - l1])], 1)
                entities_att = entities_att.write(i, entities_att_temp)
                entities_rep = entities_rep.write(i, tf.matmul(tf.matmul(a1, a2), entities_att_temp))
                return (i + 1, entities_rep, entities_att)
            def condition(i, entities_rep, entities_att):
                return i < batch_size
            _, entities_rep_final, entities_att_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, entities_rep, entities_att))
            self.entities_atts = tf.reshape(entities_att_final.stack(), [-1, self.max_entities_len])
            self.entities_reps = tf.reshape(entities_rep_final.stack(), [-1, self.n_hidden])
            
            context_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter = context_outputs_iter.unstack(context_outputs)
            entities1_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            entities1_avg_iter = entities1_avg_iter.unstack(entities1_avg)
            entities2_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            entities2_avg_iter = entities2_avg_iter.unstack(entities2_avg)
            context_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            context_lens_iter = context_lens_iter.unstack(self.context_lens)
            context_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att = tf.TensorArray(size=batch_size, dtype=tf.float32)
            def body(i, context_rep, context_att):
                a = context_outputs_iter.read(i)
                b1 = entities1_avg_iter.read(i)
                b2 = entities2_avg_iter.read(i)
                l = math_ops.to_int32(context_lens_iter.read(i))
                b12 = b1 + b2
                tf.matmul(a, weights['context_score'])
                context_score = tf.reshape(tf.nn.tanh(tf.matmul(tf.matmul(a, weights['context_score']), b12) + biases['context_score']), [1, -1])
                context_att_temp = tf.concat([tf.nn.softmax(tf.slice(context_score, [0, 0], [1, l])), tf.zeros([1, self.max_context_len - l])], 1)
                context_att = context_att.write(i, context_att_temp)
                context_rep = context_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_rep, context_att)
            def condition(i, context_rep, context_att):
                return i < batch_size
            _, context_rep_final, context_att_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, context_rep, context_att))
            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.max_context_len])
            self.context_reps = tf.reshape(context_rep_final.stack(), [-1, self.n_hidden])
            
            self.reps = tf.concat([self.entities_reps, self.context_reps], 1)
            self.predict = tf.matmul(self.reps, weights['softmax']) + biases['softmax']

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.predict, labels = self.labels))
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))
            
        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        timestamp = str(int(time.time()))
        _dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def train(self, data):
        entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens = data
        cost, cnt = 0., 0

        for sample, num in self.get_batch_data(entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens, self.batch_size, True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op], feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.test(data)
        return cost / cnt, train_acc

    def test(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, acc, cnt = 0., 0, 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(data), False, 1.0):
            loss, accuracy, step, summary = self.sess.run([self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)
        return cost / cnt, acc / cnt

    def analysis(self, train_data, test_data):
        timestamp = str(int(time.time()))

        entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens = train_data
        with open('analysis/train_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens, len(train_data), False, 1.0):
                entities_atts, context_atts, correct_pred = self.sess.run([self.entities_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(entities_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing training data')

        entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens = test_data
        pred = []
        gold = []
        labels_list = list(labels)
        with open('analysis/test_' + str(timestamp) + '.txt', 'w') as f:
            k = 0
            for sample, num in self.get_batch_data(entities1, entities2, contexts, labels, entities1_lens, entities1_lens, context_lens, len(test_data), False, 1.0):
                entities_atts, context_atts, correct_pred = self.sess.run([self.entities_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c, l in zip(entities_atts, context_atts, correct_pred, labels_list[k * num: (k + 1) * num]):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))

                    if c:
                        if (l[0] == 1):
                            pred.append(0)
                            gold.append(0)
                        else:
                            pred.append(1)
                            gold.append(1)
                    else:
                        if(l[0] == 0):
                            pred.append(1)
                            gold.append(0)
                        else:
                            pred.append(0)
                            gold.append(1)
                k += 1

        print(classification_report(gold, pred, digits=3))
        print(metrics.precision_score(gold, pred, average='macro'))
        print(metrics.recall_score(gold, pred, average='macro'))
        print(metrics.f1_score(gold, pred, average='macro'))

        out = open("../output/result.txt", "a")
        out.write(classification_report(gold, pred, digits=3))
        out.write(str(metrics.precision_score(gold, pred, average='macro')) + "\n")
        out.write(str(metrics.recall_score(gold, pred, average='macro')) + "\n")
        out.write(str(metrics.f1_score(gold, pred, average='macro')) + "\n\n")
        out.close()

        out1 = open("../output/result_num.txt", "a")
        out1.write(str(metrics.precision_score(gold, pred, average='macro')) + "\t" + str(metrics.recall_score(gold, pred, average='macro')) + "\t" +\
                                                                                   str(metrics.f1_score(gold, pred, average='macro')) + "\n")
        out1.close()

        out = open("../output/result_all_data.txt", "a")
        result = ""
        for g in gold:
            result += str(g) + " "
        out.write(result + "\n")
        result = ""
        for p in pred:
            result += str(p) + " "
        out.write(result + "\n")
        print('Finishing analyzing testing data')
    
    def run(self, train_data, test_data):
        saver = tf.train.Saver(tf.trainable_variables())

        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models/model_iter', global_step=step)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; train-loss=%.6f; test-acc=%.6f;' % (str(i), train_loss, train_acc, test_loss, test_acc))
        saver.save(self.sess, 'models/model_final')
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))

        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
        self.analysis(train_data, test_data)

    def get_batch_data(self, entities1, entities2, contexts, labels, entities1_lens, entities2_lens, context_lens, batch_size, is_shuffle, keep_prob):
        for index in get_batch_index(len(entities1), batch_size, is_shuffle):
            feed_dict = {
                self.entities1: entities1[index],
                self.entities2: entities2[index],
                self.contexts: contexts[index],
                self.labels: labels[index],
                self.entities1_lens: entities1_lens[index],
                self.entities2_lens: entities2_lens[index],
                self.context_lens: context_lens[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)
