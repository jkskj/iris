import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import time
from sklearn import preprocessing as pp


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def model(x, thera):
    return sigmod(np.dot(x, thera.T))


def cost(x, y, theta):
    left = np.multiply(-y, np.log(model(x, theta)))
    right = np.multiply(1 - y, np.log(1 - model(x, theta)))
    return np.sum(left - right) / len(x)


def gradient(x, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(x, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, x[:, j])
        grad[0, j] = np.sum(term) / len(x)
    return grad


# 三种停止策略
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    # 根据迭代次数，固定最多多少次迭代次数
    if type == STOP_ITER:
        return value > threshold
    # 根据损失值，损失值不怎么变了，停止
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    # 根据梯度变化的大小，如果梯度变化贼小贼小了，基本认为可以停止了
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


# 洗牌，乱序数据，打乱数据顺序
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return x, y


# 三种梯度下降方式，
# batchSize=1,为随机梯度下降，
# batchSize=数据总样本数，为梯度下降，
# 1<batchSize<数据总样本数,为mini_batch梯度下降

# stopType：停止策略

# thresh  ： 策略对应的阈值
def descent(data, thera, batchSize, stopType, thresh, alpha):
    # 梯度下降求解

    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    x, y = shuffleData(data)
    grad = np.zeros(thera.shape)  # 计算的梯度
    costs = [cost(x, y, thera)]  # 损失值
    while True:
        grad = gradient(x[k:k + batchSize], y[k:k + batchSize], thera)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            x, y = shuffleData(data)  # 重新洗牌
        thera = thera - alpha * grad  # 参数更新
        costs.append(cost(x, y, thera))  # 计算新的损失
        i += 1
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh): break
    return thera, i - 1, costs, grad, time.time() - init_time


# 根据实际所给的数据量，去选择初始化方式，停止策略，梯度下降方式，执行更新参数
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    # plt.show()
    return theta


# 二分类
def predict(x, thera):
    return [1 if y >= 0.5 else 0 for y in model(x, thera)]


# 多分类
def predict_plus(x, thera1, thera2, thera3):
    total = []
    for p in range(n):
        y1 = model(x, thera1)[p, 0]
        y2 = model(x, thera2)[p, 0]
        y3 = model(x, thera3)[p, 0]
        total.append(np.argmax([y1, y2, y3]))
    # print(total)
    return total


column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
# print(df_iris.head())
# print(df_iris.shape)
df_iris.insert(0, "Ones", 1)
orig_data = df_iris.values
cols = orig_data.shape[1]
x = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
for l in range(0, len(y)):
    if y[l] == "Iris-setosa":
        orig_data[l, cols - 1:cols] = 0
    elif y[l] == 'Iris-versicolor':
        orig_data[l, cols - 1:cols] = 1
    elif y[l] == 'Iris-virginica':
        orig_data[l, cols - 1:cols] = 2
# print(orig_data)
# y = orig_data[:, cols - 1:cols]
theta = np.zeros([1, 5])
# print(x[:5])
# print(y[:5])
# AttributeError: 'float' object has no attribute 'exp'
# x = np.array(x, dtype=np.float64)
# 测试损失函数
# print(cost(x, y, theta))
# 选择的梯度下降方法是基于所有样本的
n = df_iris.shape[0]
# print(n)
orig_data = np.array(orig_data, dtype=np.float64)
# 三种都尝试
# runExpe(orig_data, theta, n, STOP_ITER, thresh=200000, alpha=0.001)
# runExpe(orig_data, theta, n, STOP_COST, thresh=0.0000001, alpha=0.001)
# runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.005, alpha=0.001)
# 标准化 数据缩放
scaled_data = orig_data.copy()
scaled_data[:, 1:5] = pp.scale(orig_data[:, 1:5])
scaled_data1 = scaled_data.copy()
scaled_data2 = scaled_data.copy()
scaled_data3 = scaled_data.copy()
for m in range(0, n):
    if scaled_data1[m, 5] == 0:
        scaled_data1[m, 5] = 1
    else:
        scaled_data1[m, 5] = 0
    if scaled_data2[m, 5] != 1:
        scaled_data2[m, 5] = 0
    if scaled_data3[m, 5] == 2:
        scaled_data3[m, 5] = 1
    else:
        scaled_data3[m, 5] = 0
theta1 = runExpe(scaled_data1, theta, n, STOP_COST, thresh=0.0000001, alpha=0.001)
theta2 = runExpe(scaled_data2, theta, n, STOP_GRAD, thresh=0.005, alpha=0.01)
theta3 = runExpe(scaled_data3, theta, n, STOP_COST, thresh=0.0000001, alpha=0.01)
scaled_X = scaled_data[:, :5]
y = scaled_data[:, 5]
predictions = predict_plus(scaled_X, theta1, theta2, theta3)
correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) / (len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))
