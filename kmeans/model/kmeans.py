import math

import numpy as np
import random
from utils.draw import draw_scatter


def get_data_div(belong_list, data, k):
    div_data = []
    for label in range(0, k):
        div_data.append(data[belong_list == label])
    return div_data


def _relabel(data, belong_list, means, je):
    delta_je = 0.0
    belong_size = np.bincount(belong_list)

    for index, sample in enumerate(data):
        label = belong_list[index]
        if belong_size[label] <= 1:
            continue
        zou = np.array(list(map(lambda x: sum(x * x), means - sample)))
        for i in range(0, zou.size):
            zou[i] *= belong_size[i] / (belong_size[i] - 1) if i == label else belong_size[i] / (belong_size[i] + 1)
        mov_k = np.argmin(zou)
        if mov_k == label:
            continue
        cont = belong_size[label] / (belong_size[label] - 1)
        desc = sum((sample - means[label]) * (sample - means[label])) * cont
        je[label] -= desc
        cont = belong_size[mov_k] / (belong_size[mov_k] + 1)
        asc = sum((sample - means[mov_k]) * (sample - means[mov_k])) * cont
        je[mov_k] += asc
        delta_je += asc - desc

        means[label] += (means[label] - sample) / (belong_size[label] - 1)
        means[mov_k] += (sample - means[mov_k]) / (belong_size[mov_k] + 1)

        belong_list[index] = mov_k
        belong_size[mov_k] += 1
        belong_size[label] -= 1
    return delta_je


class Kmeans:

    def __init__(self, k):
        super(Kmeans, self).__init__()
        self.k = k
        self.model = None

    def _init_div_data(self, data):
        data_size = data.shape[0]
        means = data[random.sample(range(0, data_size), self.k)]
        div_data = []
        je = np.zeros(shape=(self.k,))
        belong_list = []
        for sample in data:
            belong = np.argmin(np.array(list(map(lambda a: sum(a * a), means - sample))))
            belong_list.append(belong)
        belong_list = np.array(belong_list)

        for label in range(0, self.k):
            div_data.append(data[belong_list == label])

        for label, cur_data in enumerate(div_data):
            cur_mean = sum(cur_data) / len(cur_data)
            means[label] = cur_mean
            je[label] = sum(list(map(lambda x: sum(x * x), cur_data - cur_mean)))

        return belong_list, means, je

    def train(self, data):
        belong_list, means, je = self._init_div_data(data)
        # draw_scatter(_get_data_div(belong_list, data, self.k), means)
        stop_flag = False
        while not stop_flag:
            delta_je = _relabel(data, belong_list, means, je)
            stop_flag = delta_je == 0.0
        # draw_scatter(_get_data_div(belong_list, data, self.k), means)
        return belong_list, means



