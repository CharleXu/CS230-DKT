# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:41:13 2021

@author: NuT
"""

import os
import pandas as pd
import numpy as np
import random


from keras.models import Model
from keras.layers import Dense, Input, LSTM, BatchNormalization, Concatenate, Lambda, Reshape, Masking, Dropout
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import pickle

def split_into_seq(df, Tx):
    if len(df.index) < Tx+1:
        # with padding
        df_tmp = df.iloc[:, 3:].to_numpy()
        return (np.pad(df_tmp, ((0, Tx+1-df_tmp.shape[0]), (0, 0))),)
    else:
        extra_row = len(df.index) % (Tx+1)
        if extra_row > 0:
            df_tmp = pd.concat([df.iloc[0: len(df.index) - extra_row, :], df.iloc[len(df.index)-(Tx+1): len(df.index), :]])
        else:
            df_tmp = df
        arr_tmp = df_tmp.iloc[:, 3:].to_numpy()
        return(np.array_split(arr_tmp, arr_tmp.shape[0]/(Tx + 1)))

def build_data_ndarray(data_processed_user, Tx):
    data_sequence = list(map(lambda x: split_into_seq(x, Tx), data_processed_user))
    data_sequence = list(filter(lambda x: x is not None, data_sequence))
    data_sequence = [item for items in data_sequence for item in items]
    data_sequence = np.stack(data_sequence)
    return data_sequence

class DataGenerator(object):
    def __init__(self, file_name, num_seq):
        """

        :param file_name: full path of dataset
        :param num_seq: number of sequence
        """
        self.filename = file_name
        self.train_seq = []
        self.dev_seq = []
        self.test_seq = []
        self.Tx = num_seq

    def read_data(self):
        # read data and sorted by student_id
        data = pd.read_csv(self.filename).sort_values('user_id')
        # group by user_id
        data_by_user = [x[1] for x in data.groupby('user_id')]
        seqs_by_student = list(map(lambda x: split_into_seq(x, self.Tx), data_by_user))
        seqs_by_student = list(filter(lambda x: x is not None, seqs_by_student))
        seqs_by_student = [item for items in seqs_by_student for item in items]
        seqs_by_student = np.stack(seqs_by_student)
        return seqs_by_student

    def split_data(self, random_seed=1):
        """

        :param seqs_by_student:
        :param random_seed:
        :return:
        """
        random.seed(random_seed)
        seqs_by_student = self.read_data()
        random.shuffle(seqs_by_student)
        n = seqs_by_student.shape[0]

        split_1 = int(0.9 * n)
        split_2 = int(0.95 * n)
        self.train_seq = seqs_by_student[:split_1]
        self.dev_seq = seqs_by_student[split_1:split_2]
        self.test_seq = seqs_by_student[split_2:]

def DKT_model(Tx, n_a, embedding_cell):
    """
    break down X into :
        - correct_1: correctness t
        - s_1:  skill vector t
        - other_categorical: other one-hot meta
        - other_numeric: other numerical meta
        - s_2: skill vector t + 1
        
    
    1. embed s_1, s_2
    2. normalize other_categorical
    3. new X: [c, s1, o1, o2]
    4. a2 = LSTM_layer(new_X, c1, a1)
    5. y' = sigmoid_layer([a2, s2])
    
    loss = binary cross entropy
    
    """
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, 254))
    
    # Define the initial hidden state a0 and initial cell state c0
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    
    # pre-process X
    s_1 = embedding_cell(Lambda(lambda z: z[:, :, 1: 124], name='S_1')(X))      # embed the skill vector for t
    s_2 = embedding_cell(Lambda(lambda z: z[:, :, 131: 254], name='S_2')(X))    # embed the skill vector for t + 1
    correct_1 = Reshape((Tx, 1))(Lambda(lambda z: z[:, :, 0], name='correctness')(X))
    other_categorical = Lambda(lambda z: z[:, :, 124:129])(X)
    other_numeric = BatchNormalization(name = 'bn0')(Lambda(lambda z: z[:, :, 129:131], name='other_num')(X))
    
    X_new = Concatenate(axis = -1)([correct_1, s_1, other_categorical, other_numeric])
    X_new = Masking(mask_value=0)(X_new)
    X_new = Dropout(rate = 0.1)(X_new)
    s_2 = Masking(mask_value=0)(s_2)
    
    a = LSTM(n_a, return_sequences = True, name = 'LSTM_1')(inputs=X_new, initial_state=[a0, c0])
    a_reg = Concatenate(axis = -1)([a, s_2])
    a_reg = Dropout(rate = 0.5)(a_reg)
    out = Dense(units=1, activation="sigmoid", name='output')(a_reg)

    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=out)
    
    return model

def main_model(Tx, embedded_size, n_a, data_file, epoch=1000):
    
    # hyperparameters
    print('>>>> Hyperparameter Set')
    print('length of the sequence: {}'.format(Tx))
    print('number of dimensions for the embedding output: {}'.format(embedded_size))
    print('number of dimensions for the hidden state of each LSTM cell: {}'.format(n_a))
    print('...Spliting Data...')
    data = DataGenerator(data_file, Tx)
    data.split_data(100)
    print(f"Train shape:{data.train_seq.shape}")
    print(f"Dev shape:{data.dev_seq.shape}")
    print(f"Test shape:{data.test_seq.shape}")
    
    Y_train = data.train_seq[:, 1:, [0]]
    X_train = np.concatenate([data.train_seq[:, 0:Tx, :], data.train_seq[:, 1:, 1:124]], axis = 2)
    
    Y_dev = data.dev_seq[:, 1:, [0]]
    X_dev = np.concatenate([data.dev_seq[:, 0:Tx, :], data.dev_seq[:, 1:, 1:124]], axis = 2)
    
    Y_test = data.test_seq[:, 1:, [0]]
    X_test = np.concatenate([data.test_seq[:, 0:Tx, :], data.test_seq[:, 1:, 1:124]], axis = 2)
    m = X_train.shape[0]
    print("Number of training examples : {}".format(m))
    
    print('...Building Model...')
    embedding_cell = Dense(units=embedded_size, activation=None, name='embedding')
    model = DKT_model(Tx, n_a, embedding_cell)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(), 'accuracy'])
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    a0_dev = np.zeros((X_dev.shape[0], n_a))
    c0_dev = np.zeros((X_dev.shape[0], n_a))
    print('...Fitting Model...')
    hist = model.fit([X_train, a0, c0], Y_train, epochs=epoch, validation_data=([X_dev, a0_dev, c0_dev], Y_dev))
    loss, auc, accuracy = model.evaluate(x= [X_dev, a0_dev, c0_dev], y = Y_dev)
    print(f'loss: {loss}, auc: {auc}, acc: {accuracy}')
    print('...Saving Model...')
    model.save(f'./results/{Tx}_{embedded_size}_{n_a}/DKTmodel')
    with open(f'./results/{Tx}_{embedded_size}_{n_a}/history.pickle', 'wb') as file:
        pickle.dump(hist.history, file)
    with open(f'./results/{Tx}_{embedded_size}_{n_a}/data.pickle', 'wb') as file:
        pickle.dump([X_train, Y_train, X_dev, Y_dev, X_test, Y_test], file)
        
if __name__ == "__main__":
    print('...Loading Data...')
    data_dir = './data'
    data_file = os.path.join(data_dir, 'skill_build_processed.csv')
    # parameter searching
    epoch = 2000
    for Tx in [8,16,24,32,48]:
        for embedded_size in [64, 128]:
            for n_a in [64,128,256]:
                main_model(Tx, embedded_size, n_a, data_file, epoch)

        
