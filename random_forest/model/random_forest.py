import random
from tqdm import tqdm
import numpy as np
from random_forest.model.decision_tree import Decision_Tree


class Random_Forest:
    def __init__(self, tree_num, random_atr_num, batch):
        super(Random_Forest, self).__init__()
        self.tree_num = tree_num
        self.random_atr_num = random_atr_num
        self.batch = batch
        self.trees = []

    def train(self, data, label, dtype):
        for _ in tqdm(range(0, self.tree_num)):
            batch_indexes = random.sample(range(0, label.size), self.batch)
            tree = Decision_Tree(gini_threshold=0.3, rf_atr_num=self.random_atr_num)
            tree.train(data[batch_indexes], label[batch_indexes], dtype)
            self.trees.append(tree)

    def query(self, sample):
        answer = []
        for tree in self.trees:
            answer.append(tree.query(sample))
        count = np.bincount(np.array(answer))
        return np.argmax(count)
