# MachineLearningImpl
这里是一些经典机器学习算法的实现，当前更新（ANN，朴素bayes， CART决策树、ID3决策树、随机森林、kNN）。实现基于C++或者python基本库（python使用numpy和pandas，不使用任何机器学习框架），初学学习之用。

## ANN（基于C++）
https://github.com/Kalzncc/SimpleANNModel

## 朴素bayes（基于C++）
https://github.com/Kalzncc/SimpleBayesClassifier

## CART决策树
```python
from random_forest.model.decision_tree import Decision_Tree

model = Decision_Tree(gini_threshold=0.01, rf_atr_num=-1)
# gini_threshold  基尼指数阈值，小于此阈值终止划分
# rf_atr_num      默认是-1，每次随机选取的属性数目，（用于随机森林）


model.train(data, label, dtype)
# data, label 分别是numpy数组格式的数据和对应的label，
# 而dtype是一个列表，其对应了每一维属性是离散值还是连续值
# 例如数据的数据为：(年龄，分数，性别)。其中年龄，分数为连续值，性别为离散值，则传入的dtype则为:[1,1,0]

out_label = model.query(sample)
# 输入一个样例，输出预测标签
```

## 随机森林
```python
from random_forest.model.random_forest import Random_Forest

model = Random_Forest(tree_num=10, random_atr_num=3, batch=100, gini_threshold=0.3)
# tree_num        决策树数量
# random_atr_num  每次随机选取的属性个数
# batch           每次训练随机选取的样例个数

model.train(data=data, label=label, dtype=dtype)
# data, label 分别是numpy数组格式的数据和对应的label，
# 而dtype是一个列表，其对应了每一维属性是离散值还是连续值
# 例如数据的数据为：(年龄，分数，性别)。其中年龄，分数为连续值，性别为离散值，则传入的dtype则为:[1,1,0]

out_label = model.query(sample)
# 输入一个样例，输出预测标签
```

## Kmeans算法
```python
from model.kmeans import Kmeans
from utils.draw import draw_scatter
from model.kmeans import get_data_div

model = Kmeans(k=4)
# K为分簇个数

bel, means = model.train(data)
# bel为每个数据的所属类
# means为每个簇的中心


draw_scatter(get_data_div(bel, data, 4), means)
#画散点图示例
```

这里演示聚簇示例
![image](https://user-images.githubusercontent.com/44296812/200552672-e5e3f832-c564-4e19-85be-120cfab5e58d.png)


## ID决策树和kNN算法

https://github.com/Kalzncc/ID3AndKNNImpl
