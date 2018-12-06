---
title: Support Vector Machine
date: 2018-12-06 11:29:10
categories: 机器学习
copyright: True
tags:
    - sklearn
    - SVM
    - Hands-On Machine Learning with Scikit-Learn and TensorFlow
---

# [线性SVM分类](http://coldjune.com/2018/05/22/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA-SVM/)

## 软间隔分类


```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge"))
))

svm_clf.fit(X, y)
```




    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linear_svc', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
         penalty='l2', random_state=None, tol=0.0001, verbose=0))])




```python
svm_clf.predict([[5.5, 1.7]])
```




    array([1.])



# 非线性SVM分类


```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline((
            ("poly_featrure", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("svm_clf", LinearSVC(C=10, loss="hinge"))
))

polynomial_svm_clf.fit(X, y)
```




    Pipeline(memory=None,
         steps=[('poly_featrure', PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)), ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm_clf', LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
         penalty='l2', random_state=None, tol=0.0001, verbose=0))])



## 多项式核


```python
from sklearn.svm import SVC
polu_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))

polu_kernel_svm_clf.fit(X, y)
```




    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm_clf', SVC(C=5, cache_size=200, class_weight=None, coef0=1,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='poly', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False))])



## 高斯径向基函数核


```python
rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))

rbf_kernel_svm_clf.fit(X, y)
```




    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm_clf', SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=5, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])



# SVM回归


```python
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
```




    LinearSVR(C=1.0, dual=True, epsilon=1.5, fit_intercept=True,
         intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
         random_state=None, tol=0.0001, verbose=0)




```python
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma='auto')
svm_poly_reg.fit(X, y)
```




    SVR(C=100, cache_size=200, coef0=0.0, degree=2, epsilon=0.1, gamma='auto',
      kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
