import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from matplotlib.colors import ListedColormap

column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
# print(df_iris.head())
# print(df_iris.describe())
iris = np.array(df_iris)
fig = plt.figure('Iris Data', figsize=(15, 15))
plt.suptitle("Andreson's Iris Dara Set\n(Blue->Setosa|Red->Versicolor|Green->Virginical)")
style = np.array(iris[:, 4])
colors = []
targets = []
for l in range(len(style)):
    if style[l] == 'Iris-setosa':
        colors.append('blue')
        targets.append(0)
    elif style[l] == 'Iris-versicolor':
        colors.append('red')
        targets.append(1)
    elif style[l] == 'Iris-virginica':
        colors.append('green')
        targets.append(2)
train_X, test_X = train_test_split(iris, test_size=0.3, random_state=5)
# k_range = range(1, 31)
# k_error = []
# # 循环，取k=1到k=31，查看误差效果
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
#     scores = cross_val_score(knn, iris[:,:3], targets, cv=6, scoring='accuracy')
#     k_error.append(1 - scores.mean())
#
# # 画图，x轴为k值，y值为误差值
# plt.plot(k_range, k_error)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Error')
# plt.show()
n_neighbors = 13

# 导入数据
column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
# print(df_iris.head())
# print(df_iris.describe())
iris = np.array(df_iris)
x = iris[:, 2:4]
# 只采用前两个feature,方便画图在二维平面显示
y = targets

h = 0.02
# 网格中的步长

# 创建彩色的图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# weights是KNN模型中的一个参数，上述参数介绍中有介绍，这里绘制两种权重参数下KNN的效果图
for weights in ['uniform', 'distance']:
    # 创建了一个knn分类器的实例，并拟合数据
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x, y)

    # 绘制决策边界，为此，我们将为每个分配一个颜色
    # 来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入一个彩色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练点
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.show()
