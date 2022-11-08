import random
from utils.utils import read_data
from model.kmeans import Kmeans
from utils.draw import draw_scatter
from model.kmeans import get_data_div
def generate_data():
    with open('../data/data.csv', 'w') as fd:
        for _ in range(0, 40):
            label = random.randint(0, 4)
            if label == 0:
                x = random.random()
                y = random.random()
            elif label == 1:
                x = random.random() + 2
                y = random.random() + 2
            elif label == 2:
                x = random.random() + 2
                y = random.random() - 2
            else:
                x = random.random() - 2
                y = random.random() - 2
            fd.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')
    with open('../data/type.csv', 'w') as fd:
        fd.write('1,1')


if __name__ == '__main__':
    # generate_data()
    data, label, dtype = read_data('../data/data.csv', '../data/type.csv')
    model = Kmeans(k=4)
    bel, means = model.train(data)
    draw_scatter(get_data_div(bel, data, 4), means)
    pass
