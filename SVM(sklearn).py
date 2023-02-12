from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
n = df_iris.shape[0]
iris = np.array(df_iris)
x = iris[:, :2]
X = np.array(x).astype(np.float32)
for l in range(n):
    if iris[l, 4] == "Iris-setosa":
        iris[l, 4] = 1
    else:
        iris[l, 4] = 0
y = iris[:, 4]
y = np.where(y <= 0, -1, 1)
# 线性核函数kernel=‘linear’
# 采用线性核kernel='linear’的效果和使用sklearn.svm.LinearSVC实现的效果一样，
# 但采用线性核时速度较慢，特别是对于大数据集，推荐使用线性核时使用LinearSVC
clf = SVC(kernel='linear')
clf.fit(X, y)
y_pred = clf.predict(X)
print(accuracy_score(y, y_pred))
