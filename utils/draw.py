import matplotlib.pyplot as plt
import pandas as pd

color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']


def draw_scatter(data_set, means):
    for i, datas in enumerate(data_set):
        x = datas[:, 0]
        y = datas[:, 1]
        plt.scatter(x, y, c=color[i])
    for i, mean in enumerate(means):
        plt.scatter(mean[0], mean[1], c=color[i], marker='x')
    plt.show()
