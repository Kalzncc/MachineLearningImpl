import random

import numpy as np

from model.support_vector_machine import Support_Vector_Machine
from utils.utils import read_data
from model.kernel_func import Poly_Kernel
import model.kernel_func as kf
from utils.draw import draw_div_line


def generate_data():
    with open('../data/data.csv', 'w') as fd:
        for _ in range(0, 100):
            x = random.random() * 10
            y = random.random() * 10
            if y > 0.82 * x + 3.2:
                label = 0
                fd.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')
            elif y < 0.82 * x + -1.2:
                label = 1
                fd.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')

    with open('../data/test.csv', 'w') as fd:
        for _ in range(0, 100):
            x = random.random() * 10
            y = random.random() * 10
            if y > 0.82 * x + 3.2:
                label = 0
                fd.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')
            elif y < 0.82 * x + -1.2:
                label = 1
                fd.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')


def run_svm():
    svm = Support_Vector_Machine(max_round=20, kernel_func=kf.Poly_Kernel(d=1))
    data, label, dtype = read_data("../data/data.csv", "../data/type.csv")

    svm.train(data, label)

    label = np.array([-1 if i == 0 else 1 for i in label])
    draw_div_line(data, label, min_x=-5, max_x=15, min_y=-5, max_y=15, query=svm.query, sv=svm.sv)
    t_data, t_label, dtype = read_data("../data/test.csv", "../data/type.csv")
    cnt = 0
    for x, y in zip(t_data, t_label):
        out = svm.query(x)
        print('query sample : {} - {}'.format(x, y))
        if y == 0 and out < 0:
            cnt += 1
            print(' out {} RIGHT!!!\n'.format(out))
        elif y == 1 and out > 0:
            cnt += 1
            print(' out {} RIGHT!!!\n'.format(out))
        else:
            print(' out {} WRONG!!!\n'.format(out))
    print('query done! right : {}, total : {}, acc : {}\n'.format(cnt, t_label.shape[0], cnt / t_label.shape[0]))


if __name__ == '__main__':
    generate_data()
    run_svm()
