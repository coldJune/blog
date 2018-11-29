---
title: Training Models
date: 2018-11-29 16:32:32
categories: 机器学习
copyright: True
tags:
    - sklearn
    - 回归
    - Hands-On Machine Learning with Scikit-Learn and TensorFlow
---

# [线性回归](http://coldjune.com/2018/05/25/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92/)

## [正规方程](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note2.pdf)


```python
# 构建类线性数据
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
```


![png](Training-Models/Training%20Models_2_0.png)



```python
#使用正规方程计算theta
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
```




    array([[3.97963397],
           [2.98594391]])




```python
# 使用theta进行预测
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict
```




    array([[3.97963397],
           [9.95152178]])




```python
# 画出预测函数
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
```


![png](Training-Models/Training%20Models_5_0.png)



```python
# 使用sklearn库的LinearRegression预测
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```




    (array([3.97963397]), array([[2.98594391]]))




```python
lin_reg.predict(X_new)
```




    array([[3.97963397],
           [9.95152178]])



# [梯度下降](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note2.pdf)


## [批量梯度下降](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note10.pdf)


```python
eta = 0.1#学习速率
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

theta
```




    array([[3.97963397],
           [2.98594391]])




```python
# 画出不同学习速率的拟合过程
theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


np.random.seed(42)
theta = np.random.randn(2, 1)

plt.figure(figsize=(10, 4))

plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(132)
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)

plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)

plt.show()
```


![png](Training-Models/Training%20Models_15_0.png)


## [随机梯度下降](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note10.pdf)


```python
# 实现
theta_path_bgd = []
n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i<20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)    
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta * gradients
        theta_path_bgd.append(theta)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
theta
```


![png](Training-Models/Training%20Models_17_0.png)





    array([[3.94377661],
           [2.97561711]])




```python
#使用sklearn的梯度下降
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

    e:\python\python36\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:152: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)





    SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
           eta0=0.1, fit_intercept=True, l1_ratio=0.15,
           learning_rate='invscaling', loss='squared_loss', max_iter=None,
           n_iter=50, n_iter_no_change=5, penalty=None, power_t=0.25,
           random_state=None, shuffle=True, tol=None, validation_fraction=0.1,
           verbose=0, warm_start=False)




```python
sgd_reg.intercept_, sgd_reg.coef_
```




    (array([3.96293115]), array([2.9698392]))



# [多项式回归](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note2.pdf)


```python
# 构建数据集
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()

```


![png](Training-Models/Training%20Models_21_0.png)



```python
#为特征添加指数
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
```




    array([2.15290063])




```python
X_poly[0]
```




    array([2.15290063, 4.63498111])




```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])

lin_reg.intercept_, lin_reg.coef_

```




    (array([2.28405711]), array([[0.88171323, 0.44446033]]))




![png](Training-Models/Training%20Models_24_1.png)


# [学习曲线](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note6.pdf)


```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=16)
    plt.axis([0,80,0,3])
    plt.legend(loc="upper right")


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
```


![png](Training-Models/Training%20Models_26_0.png)



```python
# 使用管道
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("sgd_reg", LinearRegression())
))
plot_learning_curves(polynomial_regression, X, y)
```


![png](Training-Models/Training%20Models_27_0.png)


# 正规化线性模型

## [岭回归](http://coldjune.com/2018/05/25/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92/#%E5%B2%AD%E5%9B%9E%E5%BD%92)


```python
# 使用sklearn的Ridge
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```




    array([[4.66784461]])




```python
sgd_reg = SGDRegressor(max_iter=300,tol=1e-3,penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```




    array([4.65328865])



## [Lasso回归](http://coldjune.com/2018/05/25/线性回归/#lasso)


```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```




    array([4.61400886])



## 弹性网络


```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```




    array([4.62211027])



## 及时停止


```python
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(warm_start=True, penalty=None, max_iter=300,
                      learning_rate="constant", eta0=0.0005, tol=1e-3)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
epoch, sgd_reg
```




    (999,
     SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
            eta0=0.0005, fit_intercept=True, l1_ratio=0.15,
            learning_rate='constant', loss='squared_loss', max_iter=300,
            n_iter=None, n_iter_no_change=5, penalty=None, power_t=0.25,
            random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
            verbose=0, warm_start=True))



# [逻辑回归](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note3.pdf)


```python
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
```




    ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']




```python
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
```


```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(X, y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False)




```python
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width(cm)", fontsize=14)
plt.ylabel("probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()
```


![png](Training-Models/Training%20Models_42_0.png)



```python
log_reg.predict([[1.7], [1.5]])
```




    array([1, 0])



# [多项逻辑回归](https://github.com/coldJune/machineLearning/blob/master/machineLearningCourseraNote/Note3.pdf)


```python
X = iris["data"][:, (2, 3)]
y = iris["target"]
```


```python
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
```




    LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='multinomial',
              n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False)




```python
softmax_reg.predict([[5,2]])
```




    array([2])




```python
softmax_reg.predict_proba([[5,2]])
```




    array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])

****
[源文件](https://github.com/coldJune/machineLearning/blob/master/handson-ml/Training%20Models.ipynb)
