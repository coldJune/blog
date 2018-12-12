---
title: Dimensionality Reduction
date: 2018-12-12 15:56:43
categories: 机器学习
copyright: True
tags:
    - sklearn
    - PCA
    - 维度约简
    - Hands-On Machine Learning with Scikit-Learn and TensorFlow
---
---


```python
import numpy as np

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
```

# [PCA](http://coldjune.com/2018/05/30/PCA/)


```python
X_centred = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centred)
c1 = V.T[:, 0]
c2 = V.T[:, 1]
```

## 降维


```python
W2 = V.T[:, :2]
X2D = X_centred.dot(W2)
X2D[:5]
```




    array([[-1.26203346, -0.42067648],
           [ 0.08001485,  0.35272239],
           [-1.17545763, -0.36085729],
           [-0.89305601,  0.30862856],
           [-0.73016287,  0.25404049]])



## 使用sklearn


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
X2D[:5]
```




    array([[ 1.26203346,  0.42067648],
           [-0.08001485, -0.35272239],
           [ 1.17545763,  0.36085729],
           [ 0.89305601, -0.30862856],
           [ 0.73016287, -0.25404049]])



## 可释方差


```python
print(pca.explained_variance_ratio_)
```

    [0.84248607 0.14631839]


## 选择正确的维度


```python
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
```


```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
```

## PCA压缩


```python
from six.moves import urllib
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


mnist = fetch_mldata("MNIST original", data_home='./datasets/')

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
```


```python
pca = PCA(n_components=154)
X_mnist_reduced = pca.fit_transform(X_train)
X_mnist_recovered = pca.fit_transform(X_mnist_reduced)
```

## 增量PCA


```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_mnist_reduced = inc_pca.transform(X_train)
```


```python
filename = "my_mnist.data"
m, n = X_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = X_train
del X_mm
```


```python
X_mm = np.memmap(filename, dtype='float32', mode='readonly', shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```




    IncrementalPCA(batch_size=525, copy=True, n_components=154, whiten=False)



## 随机PCA


```python
rnd_pca = PCA(n_components=154, svd_solver='randomized')
X_reduced = rnd_pca.fit_transform(X_train)
```

# Kernel PCA


```python
# 使用径向核函数
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
y = t > 6.9
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```

## 选取核函数和调整超参数


```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression(solver='lbfgs'))
])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
```




    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=Pipeline(memory=None,
         steps=[('kpca', KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
         fit_inverse_transform=False, gamma=None, kernel='linear',
         kernel_params=None, max_iter=None, n_components=2, n_jobs=None,
         random_state=None, remove_zero_eig=False, tol=0)), ('log_reg', LogisticRe...enalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False))]),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid=[{'kpca__gamma': array([0.03   , 0.03222, 0.03444, 0.03667, 0.03889, 0.04111, 0.04333,
           0.04556, 0.04778, 0.05   ]), 'kpca__kernel': ['rbf', 'sigmoid']}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
print(grid_search.best_params_)
```

    {'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}



```python
# 恢复
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
```


```python
from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)
```




    32.78630879576614



# LLE


```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```
****
[源文件](https://github.com/coldJune/machineLearning/blob/master/handson-ml/Dimensionality%20Reduction.ipynb)
