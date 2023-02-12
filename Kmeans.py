import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 随机初始化聚类中心
def randomly_initial_centroids(data, k):
    c_nums = np.random.randint(0, high=len(data), size=k)
    centroids = data[c_nums]
    return centroids


# 给样本点分类
def samples_classify(data, centroids, k):
    c = np.zeros(data.shape[0])
    for i in range(n):
        distances = (np.tile(data[i], (k, 1)) - centroids) ** 2
        ad_distances = distances.sum(axis=1)
        sq_distances = ad_distances ** 0.5
        min_number = np.where(sq_distances == np.min(sq_distances))
        min_number = min_number[0][0]
        c[i] = min_number
    return c


# 更新聚类中心
def renew_centroids(data, c, centroids, k):
    for j in range(k):
        index = (np.where(c == j))[0]
        centroids[j] = data[index].sum(axis=0) / len(index)
    return centroids


# K_means
def K_means(data, k, iterations):
    centroids = randomly_initial_centroids(data, k)
    for i in range(iterations):
        c = samples_classify(data, centroids, k)
        centroids = renew_centroids(data, c, centroids, k)
    return c, centroids, k


# 加载数据
column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
n = df_iris.shape[0]
origin_data = df_iris.values
c, centroids, k = K_means(origin_data[:, :4], 3, 300)
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 4 * i + (j + 1))
        if i == j:
            plt.text(0.3, 0.4, column_names[i], fontsize=15)
        else:
            plt.scatter(origin_data[:, j], origin_data[:, i], c=c, cmap='brg')
        if i == 0:
            plt.title(column_names[j])
        if j == 0:
            plt.ylabel(column_names[i])
plt.show()
