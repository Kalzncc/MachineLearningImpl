import math
import random

import numpy as np


def get_single_set_gini(data, label):
    if label.size == 0:
        return 0
    label_count = np.bincount(label)
    data_size = label.size
    return 1 - sum(list(map(lambda a: math.pow(a / data_size, 2), label_count)))


def get_multiple_set_gini(data_set, label_set):
    res = 0
    total_size = 0
    for data, label in zip(data_set, label_set):
        total_size += label.size
        res += label.size * get_single_set_gini(data, label)
    return res / total_size


def div_by_problem(data, problem):
    if problem[1] == '<':
        div_bool = data[:, problem[0]] < problem[2]
        div_not_bool = data[:, problem[0]] >= problem[2]
    else:
        div_bool = data[:, problem[0]] == problem[2]
        div_not_bool = data[:, problem[0]] != problem[2]
    return div_bool, div_not_bool


def _generate_problem(data, dtype):
    atr_size = len(dtype)
    problem = []
    for column in range(0, atr_size):
        unique_val = np.unique(data[:, column])
        problem.extend([(column, '=' if dtype[column] == 0 else '<', val) for val in unique_val])
    return problem


def execute_problem(data, problem):
    if problem[1] == '=':
        return 0 if data[problem[0]] == problem[2] else 1
    else:
        return 0 if data[problem[0]] < problem[2] else 1


class Decision_Tree:

    def __init__(self, gini_threshold=0.01, rf_atr_num=-1):
        super(Decision_Tree, self).__init__()
        self.tree = None
        self.random_forest_atr_num = rf_atr_num
        self.gini_threshold = gini_threshold

    def _train(self, data, label, problem):
        if get_single_set_gini(data, label) < self.gini_threshold:
            node = {'label': np.argmax(np.bincount(label)), 'type': 'leaf'}
            return node
        if self.random_forest_atr_num == -1:
            choice_atr = range(0, data.shape[1])
        else:
            choice_atr = random.sample(range(0, data.shape[1]), self.random_forest_atr_num)
        choice_atr = set(choice_atr)
        node = {'type': 'non-leaf'}
        mi_gini = None
        mi_index = -1
        for index, value in enumerate(problem):
            if value[0] not in choice_atr:
                continue
            div_bool, div_not_bool = div_by_problem(data, value)
            data_set = []
            label_set = []
            data_set.append(data[div_bool])
            label_set.append(label[div_bool])
            data_set.append(data[div_not_bool])
            label_set.append(label[div_not_bool])
            cur_gini = get_multiple_set_gini(data_set, label_set)
            if mi_gini is None or cur_gini < mi_gini:
                mi_gini = cur_gini
                mi_index = index
        mi_div_bool, mi_div_not_bool = div_by_problem(data, problem[mi_index])
        node['problem'] = problem[mi_index]
        node['sub_node'] = []
        node['sub_node'].append(self._train(data[mi_div_bool], label[mi_div_bool], problem))
        node['sub_node'].append(self._train(data[mi_div_not_bool], label[mi_div_not_bool], problem))
        return node

    def _query(self, node, sample):
        if node['type'] == 'leaf':
            return node['label']
        problem = node['problem']
        return self._query(node['sub_node'][execute_problem(sample, problem)], sample)

    def train(self, data, label, dtype):
        problem = _generate_problem(data, dtype)
        self.tree = self._train(data, label, problem)

    def query(self, sample):
        if self.tree is None:
            raise "No Train"
        return self._query(self.tree, sample)
