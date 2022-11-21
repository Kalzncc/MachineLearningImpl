# MachineLearningImpl
这里是一些经典机器学习算法的实现。实现基于C++或者python基本库（python使用numpy和pandas，不使用任何机器学习框架），初学学习之用。

当前更新：
> <a href="#ann">ANN</a> \
> <a href="#bys">朴素bayes</a>  
> <a href="#cart">CART决策树</a> \
> <a href="#id3">ID3决策树 </a>\
> <a href="#rf">随机森林 </a>\
> <a href="#knn">kNN </a>\
> <a href="#kmeans">Kmeans </a> \
> <a href="#svm">SVM</a>

<a id="ann"/>

## ANN（基于C++）
https://github.com/Kalzncc/SimpleANNModel

<a id="bys"/>

## 朴素bayes（基于C++）
https://github.com/Kalzncc/SimpleBayesClassifier

<a id="cart"/>

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

<a id="rf"/>

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

<a id="kmeans"/>

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


<a id="svm"/>

## SVM 基于SMO算法

公式推定：https://kalzncc.github.io/2022/11/19/smo/

这里吐槽一下，这个东西是真的难啊。一度想要放弃，不过最后还是推出来并实现出来了。

```python
from model.support_vector_machine import Support_Vector_Machine
from utils.utils import read_data
import model.kernel_func as kf
from utils.draw import draw_div_line

svm = Support_Vector_Machine(max_round, delta_alpha_threshold, c, kkt_tolerance, kernel_func)
# max_round             最大优化轮次
# delta_alpha_threshold 优化alpha时的最小变化阈值（变化小于此阈值时视为优化失败）
# c                     软间隔松弛参数，默认极大值
# kkt_tolerance         kkt条件容忍程度阈值（辨别时依据此阈值判定一变量是否满足kkt）
# kernel_func           核函数， 在model.kernel_func.py中定义了一些核函数示例，可以参考。默认为线性核

svm.train(data, label)
# data, label 分别是numpy数组格式的数据和对应的label，应为二分类，label只有0，1两种标签

out_f = svm.query(sample)
# 询问样例标签，输出为一个数值，为原f函数输出，如果值小于0应划分至-1，如果值大于0应划分至1

label = np.array([-1 if i == 0 else 1 for i in label])
draw_div_line(data, label, min_x=-5, max_x=15, min_y=-5, max_y=15, query=svm.query, sv=svm.sv)
# 画分割线图表
```
有关核函数，在<a href="https://github.com/Kalzncc/MachineLearningImpl/blob/master/svm/model/kernel_func.py">model.kernel_func.py</a>有示例，核函数需要定义run方法，传入两个等维度numpy向量，输出一个标量值。这里示例创建一个sigma为1.0的高斯核函数。
```python
# main.py
from svm.model.kernel_func import Gauss_Kernel
svm = Support_Vector_Machine(max_round=10, kernel_func=Gauss_Kernel(sigma=1.0))

# svm.model.kernel_func.py
class Gauss_Kernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def run(self, x, y):
        return math.exp(-sum((x - y) * (x - y)) / (2 * self.sigma * self.sigma))
```
下面是示例，这里带有红色x的样本是支持向量

![image](https://user-images.githubusercontent.com/44296812/201329434-c301b4be-f906-4f0c-851f-c66081fae2ce.png)



<a id="id3"/>
<a id="knn"/>

## ID3决策树和kNN算法

https://github.com/Kalzncc/ID3AndKNNImpl
