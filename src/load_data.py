import numpy as np
import pandas as pd
import random


def split_into_seq(df, Tx):
    if len(df.index) < Tx+1:
        return None
    else:
        extra_row = len(df.index) % (Tx+1)
        if extra_row > 0:
            df_tmp = pd.concat([df.iloc[0: len(df.index) - extra_row, :], df.iloc[len(df.index)-(Tx+1): len(df.index), :]])
        else:
            df_tmp = df
        arr_tmp = df_tmp.iloc[:, 3:].to_numpy()
        return np.array_split(arr_tmp, arr_tmp.shape[0] / (Tx + 1))


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
        random_seed(random_seed)
        seqs_by_student = self.read_data()
        random.shuffle(seqs_by_student)
        n = seqs_by_student.shape[0]

        split_1 = int(0.9 * n)
        split_2 = int(0.95 * n)
        self.train_seq = seqs_by_student[:split_1]
        self.dev_seq = seqs_by_student[split_1:split_2]
        self.test_seq = seqs_by_student[split_2:]


if __name__ == "__main__":
    file = 'skill_build_processed.csv'
    Tx = 16
    data = DataGenerator(file, Tx)
    data.split_data()
