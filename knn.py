import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import operator


def knn(in_x, x_labels, y_labels, k):
    x_labels_size = x_labels.shape[0]
    distances = (np.tile(in_x, (x_labels_size, 1)) - x_labels) ** 2
    ad_distances = distances.sum(axis=1)
    sq_distances = ad_distances ** 0.5
    ed_distances = sq_distances.argsort()
    class_dict = {}
    for i in range(k):
        label = y_labels[ed_distances[i]]
        class_dict[label] = class_dict.get(label, 0) + 1
    sort_dict = sorted(class_dict.items(), key=operator.itemgetter(1), reverse=True)
    # print(sort_dict[0][0])
    return sort_dict[0][0]


column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
iris = np.array(df_iris)
fig = plt.figure('Iris Data', figsize=(15, 15))
plt.suptitle("Andreson's Iris Dara Set\n(Blue->Setosa|Red->Versicolor|Green->Virginical)")
style = np.array(iris[:, 4])
colors = []
for l in range(len(style)):
    if style[l] == 'Iris-setosa':
        colors.append('blue')
    elif style[l] == 'Iris-versicolor':
        colors.append('red')
    elif style[l] == 'Iris-virginica':
        colors.append('green')
train_X, test_X = train_test_split(iris, test_size=0.5, random_state=0)
n = 0
for j in range(test_X.shape[0]):
    res = knn(test_X[j, :4], train_X[:, :4], train_X[:, 4], 12)
    # print(test_X[j,4])
    if res == test_X[j, 4]:
        n += 1
accuracy = n / j
print('accuracy = {0}%'.format(accuracy * 100))
