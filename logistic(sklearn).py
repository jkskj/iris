import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
df_iris = pd.read_csv(filepath_or_buffer='iris.data', header=0, names=column_names)
iris = np.array(df_iris)
style = np.array(iris[:, 4])
targets = []
for l in range(len(style)):
    if style[l] == 'Iris-setosa':
        targets.append(0)
    elif style[l] == 'Iris-versicolor':
        targets.append(1)
    elif style[l] == 'Iris-virginica':
        targets.append(2)
X_train, X_test, y_train, y_test = train_test_split(iris[:, :4], targets, test_size=0.2, random_state=3)
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 查看其对应的w，3分类包含三个逻辑回归的参数
print('the weight(w) of Logistic Regression:\n', lr.coef_)
# 查看其对应的w0
print('the intercept(w0) of Logistic Regression:\n', lr.intercept_)

# 预测
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy on train set is:', metrics.accuracy_score(y_train, y_train_pred))
print('The accuracy on test set is:', metrics.accuracy_score(y_test, y_test_pred))

# 绘制混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(y_test, y_test_pred)
print('The confusion matrix result:\n', confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
