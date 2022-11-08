import math
import random
from utils.utils import read_data
from random_forest.model.decision_tree import Decision_Tree
from random_forest.model.random_forest import Random_Forest
import numpy as np
import numpy.random

def generate_data():
    with open('../data/data.csv', 'w') as fd:
        fd.write('\n')
        for i in range(0, 100):
            x = random.random() * 2
            y = random.random() * 2
            if x < 1 and y < 1:
                label = 1
            elif x > 1:
                label = 2
            else:
                label = 3
            fd.write(str(x)+','+str(y)+','+str(label)+'\n')
    with open('../data/type.csv', 'w') as fd:
        fd.write('1,1')
    with open('../data/test.csv', 'w') as fd:
        fd.write('\n')
        for i in range(0, 40):
            x = random.random() * 2
            y = random.random() * 2
            if x < 1 and y < 1:
                label = 1
            elif x > 1:
                label = 2
            else:
                label = 3
            fd.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')

def generate_data2():
    with open('../data/data.csv', 'w') as fd:
        fd.write('\n')
        for _ in range(0, 1000):
            data = np.random.rand(6)
            dif = sum(list(map(lambda a: math.pow(a, 2), data)))
            dif = math.sqrt(dif)
            if dif < 1.4 and math.pow(data[0] + data[1], 2) < 0.5:
                label = 1
            elif dif < 1.4 and math.pow(data[0] + data[1], 2) > 0.5:
                label = 2
            elif math.pow(data[0] + data[1], 2) < 0.5:
                label = 3
            else:
                label = 4
            noise = np.random.rand(6) * 0.001
            data = data + noise
            s = ','.join(map(str, data))
            s += ',' + str(label)
            fd.write(s+'\n')
    with open('../data/test.csv', 'w') as fd:
        fd.write('\n')
        for _ in range(0, 100):
            data = np.random.rand(6)
            dif = sum(list(map(lambda a: math.pow(a, 2), data)))
            dif = math.sqrt(dif)
            if dif < 1.4 and math.pow(data[0] + data[1], 2) < 0.5:
                label = 1
            elif dif < 1.4 and math.pow(data[0] + data[1], 2) > 0.5:
                label = 2
            elif math.pow(data[0] + data[1], 2) < 0.5:
                label = 3
            else:
                label = 4
            noise = np.random.rand(6) * 0.01
            data = data + noise
            s = ','.join(map(str, data))
            s += ',' + str(label)
            fd.write(s + '\n')
    with open('../data/type.csv', 'w') as fd:
        fd.write('1,1,1,1,1,1')

def test_decision_tree():
    data, label, dtype = read_data('../data/data.csv', 'data/type.csv')
    test_data, test_label, test_dtype = read_data('../data/test.csv', 'data/type.csv');
    model = Decision_Tree(gini_threshold=0.01, rf_atr_num=1)
    model.train(data, label, dtype)
    cnt = 0
    for index, value in enumerate(test_data):
        out = model.query(value)
        if out == test_label[index]:
            cnt += 1
    print('Query Done : ', cnt, '/', test_label.size)


def test_random_forest():
    data, label, dtype = read_data('../data/data.csv', 'data/type.csv')
    test_data, test_label, test_dtype = read_data('../data/test.csv', 'data/type.csv');
    model = Random_Forest(tree_num=10, random_atr_num=3, batch=100)
    model.train(data=data, label=label, dtype=dtype)
    cnt = 0
    for index, value in enumerate(test_data):
        out = model.query(value)
        if out == test_label[index]:
            cnt += 1
    print('Query Done : ', cnt, '/', test_label.size)


if __name__ == '__main__':
    generate_data2()
    test_random_forest()
