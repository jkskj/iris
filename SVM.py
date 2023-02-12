import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradient1(w, X, y, b, lr):
    for i in range(1):
        for idx, x_i in enumerate(X):
            y_i = y[idx]
            cond = y_i * (np.dot(x_i, w) - b) >= 1
            if cond:
                w -= lr * 3 * w
            else:
                w -= lr * (2 * w - np.dot(x_i, y_i))
                b -= lr * y_i
    return w, b


def predict1(X, w, b):
    pred = np.dot(X, w) - b
    return np.sign(pred)


column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
n = df_iris.shape[0]
np.random.seed(12)
num = 50
x1 = np.random.multivariate_normal([0, 0], ([1, .75], [.75, 1]), num)
x2 = np.random.multivariate_normal([1, 4], ([1, .75], [.75, 1]), num)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num), np.ones(num)))
y = np.where(y <= 0, -1, 1)
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.6)
plt.show()
w, b, lr = np.random.random(X.shape[1]), 0, 0.0001
w, b = gradient1(w, X, y, b, lr)
svm_pred = predict1(X, w, b)
correct = 0
for i in range(len(svm_pred)):
    if svm_pred[i] == y[i]:
        correct += 1
accuracy = correct / len(svm_pred)
print('accuracy = {0}%'.format(accuracy * 100))

iris = np.array(df_iris)
x = iris[:, :2]
X = np.array(x).astype(np.float32)
for l in range(n):
    if iris[l, 4] == "Iris-setosa":
        iris[l, 4] = 0
    else:
        iris[l, 4] = 1
y = iris[:, 4]
y = np.where(y <= 0, -1, 1)
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.6)
plt.show()
w, b, lr = np.random.random(X.shape[1]), 0, 0.0001
w, b = gradient1(w, X, y, b, lr)
svm_pred = predict1(X, w, b)
correct = 0
for i in range(len(svm_pred)):
    if svm_pred[i] == y[i]:
        correct += 1
accuracy = correct / len(svm_pred)
print('accuracy = {0}%'.format(accuracy * 100))
