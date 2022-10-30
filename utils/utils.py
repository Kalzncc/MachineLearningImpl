import numpy as np
import pandas as pd


def read_data(type_path, csv_path):
    csv_data = pd.read_csv(type_path)
    with open(csv_path, 'r') as type_file:
        type_data = type_file.readline()
    type_list = type_data.split(',')
    data = np.array(csv_data)
    label = data[:, -1]
    data = data[:, :-1]
    label = np.array(list(map(int, label)))
    return data, label, type_list


def read_test(csv_path):
    csv_data = pd.read_csv(csv_path)
    return np.array(csv_data)