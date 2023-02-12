import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import fowlkes_mallows_score

column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
iris = np.array(df_iris)
x = np.array(iris[:, :4], dtype=np.float64)
style = np.array(iris[:, 4])
targets = []
for l in range(len(style)):
    if style[l] == 'Iris-setosa':
        targets.append(0)
    elif style[l] == 'Iris-versicolor':
        targets.append(1)
    elif style[l] == 'Iris-virginica':
        targets.append(2)
MMS = MinMaxScaler().fit(x)
data = MMS.transform(x)
# 构建KMeans模型训练数据
cluster = KMeans(n_clusters=3, random_state=123).fit(data)
# 获取聚类结果
y_pred = cluster.labels_
# 获取质心
centers = cluster.cluster_centers_
# [[0.70726496 0.4508547  0.79704476 0.82478632]
#  [0.19557823 0.59013605 0.07886544 0.06037415]
#  [0.44125683 0.30737705 0.57571548 0.54918033]]
# 查看簇内平方和
inertia = cluster.inertia_
# 6.995764076579158
# 聚类结果可视化
# 进行数据降维处理
tsne = TSNE(n_components=2, init='random', random_state=123).fit(data)
df = pd.DataFrame(tsne.embedding_)
df['labels'] = y_pred
df1 = df[df['labels'] == 0]
df2 = df[df['labels'] == 1]
df3 = df[df['labels'] == 2]
# 绘制画布
fig = plt.figure(figsize=(9, 6))
plt.plot(df1[0], df1[1], 'bo', df2[0], df2[1], 'r*', df3[0], df3[1], 'gD')
plt.show()
# 使用轮廓系数法评价K-Means聚类模型 --- 畸变程度
silhouetteScore = []
for i in range(2, 15):
    # 构建并训练模型
    kmeans = KMeans(n_clusters=i, random_state=123).fit(data)
    score = silhouette_score(data, kmeans.labels_)
    silhouetteScore.append(score)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 15), silhouetteScore, linewidth=1.5, linestyle='-')
plt.show()
# 卡林斯基-哈拉巴斯指数
chs = []
for i in range(2, 15):
    # 构建聚类模型
    kmeans = KMeans(n_clusters=i, random_state=112).fit(data)
    chsScore = calinski_harabaz_score(data, kmeans.labels_)
    chs.append(chsScore)
plt.figure(figsize=(10, 8))
plt.plot(range(2, 15), chs, linewidth=1.5, linestyle='-')
plt.show()
# FMI评价法 --- 需要有真实标签
fms = []
for i in range(2, 15):
    # 构建聚类模型
    kmeans = KMeans(n_clusters=i, random_state=112).fit(data)
    fmsScore = fowlkes_mallows_score(targets, kmeans.labels_)
    fms.append(fmsScore)
plt.figure(figsize=(10, 8))
plt.plot(range(2, 15), fms, linewidth=1.5, linestyle='-')
plt.show()
