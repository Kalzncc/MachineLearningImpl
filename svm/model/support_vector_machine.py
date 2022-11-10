import math
import random

import svm.model.kernel_func as kf
import numpy as np
from utils.draw import draw_div_line


def _alpha_swap(a, data, label, id1, id2):
    a[id1], a[id2] = a[id2], a[id1]
    data[[id1, id2]] = data[[id2, id1]]
    label[id1], label[id2] = label[id2], label[id1]


def _init_alpha(label):
    label_size = np.bincount(label)
    a = np.zeros(shape=label.shape)
    return np.array(a)


class Support_Vector_Machine:
    def __init__(self, max_round, delta_alpha_threshold=0.000001, c=100000000000.0, kkt_tolerance=0.001,
                 kernel_func=kf.Linear_Kernel()):
        super(Support_Vector_Machine, self).__init__()
        self.kernel_func = kernel_func
        self.c = c
        self.kkt_tolerance = kkt_tolerance
        self.max_round = max_round
        self.delta_alpha_threshold = delta_alpha_threshold
        self.sv = None
        self.sv_y = None
        self.sv_a = None
        self.sv_b = None

        self.a = None
        self.b = None
        self.y = None
        self.v = None

    def f(self, a, data, label, b, x):
        return sum([a[i] * label[i] * self.kernel_func.run(data[i], x) for i in range(0, label.shape[0])]) + b

    def h(self, a, data, label, b, x, y):
        return self.f(a, data, label, b, x) * y

    def E(self, a, data, label, b, x, y):
        return self.f(a, data, label, b, x) - y

    def _select_a(self, a, b, data, label, sv_select):
        sus = False
        # 选择首个变量
        for i in range(0, label.shape[0]):
            # 是否选择边缘（非支持向量）作为第一个变量
            if sv_select and (a[i] == 0 or a[i] == self.c):
                continue
            # 不满足kkt条件时选取
            kkt_cod = label[i] * self.E(a, data, label, b[0], data[i], label[i])
            if (kkt_cod > self.kkt_tolerance and a[i] > 0) or (kkt_cod < -self.kkt_tolerance and a[i] < self.c):
                a1_index = i
            else:
                continue

            errors = [self.E(a, data, label, b[0], data[_], label[_]) for _ in range(0, label.shape[0])]

            # 第二变量，第一轮选择，从非边缘集（支持向量）中选取第二个变量
            max_delta_e = -1
            max_index = -1
            for j in range(1, label.shape[0]):
                if j == a1_index:
                    continue
                if a[j] == 0 or a[j] == self.c:
                    continue
                if max_index == -1 or max_delta_e < math.fabs(errors[a1_index] - errors[j]):
                    max_index = j
                    max_delta_e = math.fabs(errors[a1_index] - errors[j])
            a2_index = max_index

            # 选择后，执行优化，如优化成功，则继续选取，否则展开第二轮
            cur_sus = self._optimize(a, b, data, label, a1_index, a2_index)
            sus = sus or cur_sus
            if cur_sus:
                continue

            # 第二变量，第二轮选取，从全部集合中选取第二变量
            max_delta_e = -1
            max_index = -1
            for j in range(1, label.shape[0]):
                if j == a1_index:
                    continue
                if max_index == -1 or max_delta_e < math.fabs(errors[a1_index] - errors[j]):
                    max_index = j
                    max_delta_e = math.fabs(errors[a1_index] - errors[j])
            a2_index = max_index
            cur_sus = self._optimize(a, b, data, label, a1_index, a2_index)
            sus = sus or cur_sus
            if cur_sus:
                continue

            # 第二变量，第三轮选取，随机选取。
            a2_index = random.randint(0, label.shape[0] - 1)
            while a2_index == a1_index:
                a2_index = random.randint(0, label.shape[0] - 1)
            _alpha_swap(a, data, label, 1, a2_index)
            errors[1], errors[a2_index] = errors[a2_index], errors[1]
            sus = sus or self._optimize(a, b, data, label, a1_index, a2_index)

        return not sus

    def _optimize(self, a, b, data, label, i, j):
        if label[i] != label[j]:
            L = max(0, a[j] - a[i])
            H = min(self.c, self.c + a[j] - a[i])
        else:
            L = max(0, a[i] + a[j] - self.c)
            H = max(self.c, a[i] + a[j])

        eta = self.kernel_func.run(data[i], data[i]) + self.kernel_func.run(data[j], data[j]) - 2.0 * self.kernel_func.run(data[i], data[j])
        if eta <= 0:
            return False

        E0 = self.E(a, data, label, b[0], data[i], label[i])
        E1 = self.E(a, data, label, b[0], data[j], label[j])
        error = E0 - E1

        M = a[j] + label[j] * error / eta
        new_a1 = M
        if new_a1 > H:
            new_a1 = H
        elif L > new_a1:
            new_a1 = L
        delta_a1 = new_a1 - a[j]

        if delta_a1 < self.delta_alpha_threshold:
            return False

        new_a0 = a[i] - label[i] * label[j] * delta_a1
        delta_a0 = new_a0 - a[i]
        b0 = -E0 - label[i] * self.kernel_func.run(data[i], data[i]) * delta_a0 - label[j] * self.kernel_func.run(
            data[j], data[i]) * delta_a1 + b[0]
        b1 = -E1 - label[i] * self.kernel_func.run(data[i], data[j]) * delta_a0 - label[j] * self.kernel_func.run(
            data[j], data[j]) * delta_a1 + b[0]
        a[i] = new_a0
        a[j] = new_a1
        b[0] = (b0 + b1) / 2
        return True

    def pre_query(self, sample):
        wx = sum([self.a[i] * self.y[i] * self.kernel_func.run(self.v[i], sample) for i in
                  range(0, self.a.shape[0])])
        return wx + self.b

    def smo(self, a, data, label):
        stop_flag = False
        b = [0]
        cur_round = 0
        while not stop_flag and cur_round < self.max_round:
            stop_flag = self._select_a(a, b, data, label, True)
            stop_flag = self._select_a(a, b, data, label, False)
            cur_round += 1
        return b

    def train(self, o_data, o_label):
        data = o_data.copy()
        label = o_label.copy()
        a = _init_alpha(label)
        label = np.array([1 if _ == 1 else -1 for _ in label])
        b = self.smo(a, data, label)
        sv_index = np.where((a != 0) & (a != self.c))
        self.sv_a = a[sv_index]
        self.sv_b = b
        self.sv = data[sv_index]
        self.sv_y = label[sv_index]

    def query(self, sample):
        if self.sv is None:
            raise 'No train'
        wx = sum([self.sv_a[i] * self.sv_y[i] * self.kernel_func.run(self.sv[i], sample)
                  for i in range(0, self.sv_a.shape[0])])
        return wx + self.sv_b[0]
