import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']


def draw_scatter(data_set, means):
    for i, datas in enumerate(data_set):
        x = datas[:, 0]
        y = datas[:, 1]
        plt.scatter(x, y, c=color[i])
    for i, mean in enumerate(means):
        plt.scatter(mean[0], mean[1], c=color[i], marker='x')
    plt.show()


def draw_div_line(data, label, min_x, max_x, min_y, max_y, query, step=0.1, threshold=0.01, sv=None):
    data1 = data[label == -1]
    data2 = data[label == 1]
    data_set = [data1, data2]
    for i, datas in enumerate(data_set):
        x = datas[:, 0]
        y = datas[:, 1]
        plt.scatter(x, y, c=color[i])

    if sv is not None:
        sv_x = sv[:, 0]
        sv_y = sv[:, 1]
        plt.scatter(sv_x, sv_y, c=color[3], marker='x')

    if query is not None:
        xlist = []
        ylist = []
        for x in np.arange(min_x, max_x, step):
            for y in np.arange(min_y, max_y, step):
                if query([x, y]) < threshold:
                    xlist.append(x)
                    ylist.append(y)
                    break
        plt.plot(xlist, ylist)
    plt.show()
