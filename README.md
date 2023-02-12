# 对iris鸢尾花数据集的分类
我一共用了四种算法，分别为-means,KNN,Logistic,SVM,都有用sklearn库算法和不用sklearn库的两个版本。
## KNN
比较简单，就是每个样本都可以用它最接近的K个邻近值来代表。用欧式距离计算公式计算即可。
## K-means
与KNN相似，但KNN是分类算法，这是聚类算法。根据原数据知道有三类，直接分为三个聚类中心，同样计算欧式距离即可。
因为随机初始化聚类中心，导致每次结果分为三类的索引都不同，所以没有求正确率。
## Logistic回归
建立回归公式，计算概率，取最大的概率。分第一类和第三类都没问题，正确率很高，
但第二类正确率极低，我猜测是模型建的不够好，可能要高阶公式。
## SVM
巨难，到现在还似懂非懂。我没用sklearn库的是看B站的一个视频，
她用的是梯度下降，而不是SMO算法，我想把它直接套在iris身上，但失败了，有问题。
她视频里的数据是是随机生成的，效果很好，但套到iris身上就不行了，最后结果迭代小于等于10时全是1，大于时全是-1。
我也不知道是什么原因，可能是参数的原因。后面如果发现是哪里错了会修正，但不得不说SVM真的难┭┮﹏┭┮