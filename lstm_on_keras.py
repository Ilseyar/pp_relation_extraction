import gensim
import keras
import numpy as np
from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras_preprocessing.text import Tokenizer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from torch.utils.data import DataLoader

from ian_pytorch.data_utils import ABSADatesetReader

EMBEDDING_DIM = 200
max_words = 74
dataset = 'aimed'


absa_dataset = ABSADatesetReader(dataset="aimed", embed_dim=EMBEDDING_DIM, max_seq_len=max_words, fold_num=1)
x_train_data = np.asarray([x['text_raw_indices'] for x in absa_dataset.train_data])
y_train_data = keras.utils.to_categorical([x['polarity'] for x in absa_dataset.train_data], 2)
x_test_data = np.asarray([x['text_raw_indices'] for x in absa_dataset.test_data])
y_test_data = keras.utils.to_categorical([x['polarity'] for x in absa_dataset.test_data], 2)

# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(text)
#
# w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
#         "data/w2v/all.norm-sz100-w10-cb0-it1-min100.w2v", binary=True, unicode_errors='ignore')

# word_index = tokenizer.word_index
# embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if word in w2v_model:
#         embedding_vector = w2v_model[word]
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(len(absa_dataset.embedding_matrix),
                                      EMBEDDING_DIM,
                                      weights=[absa_dataset.embedding_matrix],
                                      input_length=max_words,
                                      trainable=True)
sequence_input = Input(shape=(max_words,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm = Bidirectional(LSTM(units=400, dropout=0.5))(embedded_sequences)
# dropout = Dropout(0.5)(lstm)
dense = Dense(2, activation="softmax")(lstm)

model = Model(inputs=[sequence_input],
              outputs=[dense])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train_data, axis=1)), np.argmax(y_train_data, axis=1))
model.fit(x_train_data, y_train_data,
          epochs=100,
          batch_size=16,
          # class_weight=class_weights,
          verbose=1)

preds = model.predict(x_test_data)
preds = np.argmax(preds, axis=1)
y_test = np.argmax(y_test_data, axis=1)

print(classification_report(y_test, preds, digits=3))
print(metrics.precision_score(y_test, preds, average='macro'))
print(metrics.recall_score(y_test, preds, average='macro'))
print(metrics.f1_score(y_test, preds, average='macro'))




