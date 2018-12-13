---
title: Up and Runing with TensorFlow
date: 2018-12-13 17:28:38
categories: 机器学习
copyright: True
tags:
    - TensorFlow
    - Hands-On Machine Learning with Scikit-Learn and TensorFlow
---
---

# Hello World


```python
import tensorflow as tf

# 非立即执行而是创建一个图
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```


```python
# 打开session执行
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
```

    42



```python
# 关闭session
sess.close()
```


```python
# 使用块结构运行
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)
```

    42



```python
# 使用global_variables_initialzer创建全局初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)
```

    42



```python
# 使用交互式session
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()
```

    42


# 管理图


```python
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()
```




    True




```python
# 创建独立的图
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
```


```python
x2.graph is graph
```




    True




```python
x2.graph is tf.get_default_graph()
```




    False



# 节点的生命周期


```python
w = tf.constant(3)
x = w + 2
y = x + 5
z = x + 3
```


```python
#未重复利用结果w和x
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())
```

    10
    8



```python
# 将y和z放在同一个图中以重复利用w和x
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
```

    10
    8


# 使用TensorFlow训练线性回归


```python
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
# 使用正规方程计算theta
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value[:5])
```

    [[-3.7185181e+01]
     [ 4.3633747e-01]
     [ 9.3952334e-03]
     [-1.0711310e-01]
     [ 6.4479220e-01]]


# 实现梯度下降

## 手动计算梯度


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scale_housing_data = scaler.fit_transform(housing.data)
scale_housing_data_bias = np.c_[np.ones((m, 1)), scale_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scale_housing_data_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_normal([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta-learning_rate*gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE=", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta[:5])
```

    Epoch 0 MSE= 8.3278
    Epoch 100 MSE= 0.78606343
    Epoch 200 MSE= 0.64422286
    Epoch 300 MSE= 0.6223913
    Epoch 400 MSE= 0.6058846
    Epoch 500 MSE= 0.59219676
    Epoch 600 MSE= 0.58081675
    Epoch 700 MSE= 0.57135147
    Epoch 800 MSE= 0.5634769
    Epoch 900 MSE= 0.5569242
    [[2.0685523 ]
     [0.6491688 ]
     [0.08126644]
     [0.06857597]
     [0.03242581]]


## 使用自动微分


```python
gradients = tf.gradients(mse, [theta])[0]
```

## 使用优化器


```python
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```


```python
# 动量优化器
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
```

# 输送数据


```python
# 使用占位符
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
```


```python
print(B_val_1)
```

    [[6. 7. 8.]]



```python
print(B_val_2)
```

    [[ 9. 10. 11.]
     [12. 13. 14.]]



```python
# 小批量梯度下降
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scale_housing_data_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
```


```python
best_theta
```




    array([[ 2.0685525 ],
           [ 0.8296056 ],
           [ 0.11874896],
           [-0.26550332],
           [ 0.3056771 ],
           [-0.00450377],
           [-0.03932568],
           [-0.89991784],
           [-0.87057173]], dtype=float32)



# 保存和加载模型


```python
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scale_housing_data_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_normal([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("epoch", epoch, "MSE=", mse.eval())
            save_path = saver.save(sess, "tmp/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, 'tmp/my_model_final.ckpt')
```

    epoch 0 MSE= 34.112576
    epoch 100 MSE= 1.2404147
    epoch 200 MSE= 0.62713087
    epoch 300 MSE= 0.58235973
    epoch 400 MSE= 0.5691915
    epoch 500 MSE= 0.5602364
    epoch 600 MSE= 0.55325645
    epoch 700 MSE= 0.54772204
    epoch 800 MSE= 0.54330814
    epoch 900 MSE= 0.53977317



```python
best_theta
```




    array([[ 2.0685523 ],
           [ 0.68664634],
           [ 0.10857289],
           [ 0.03570269],
           [ 0.04292491],
           [-0.00635225],
           [-0.0354162 ],
           [-1.1083173 ],
           [-1.0608274 ]], dtype=float32)




```python
# 加载模型
with tf.Session() as sess:
    saver.restore(sess, "tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval()
```

    INFO:tensorflow:Restoring parameters from tmp/my_model_final.ckpt



```python
np.allclose(best_theta, best_theta_restored)
```




    True




```python
#保存指定变量
saver = tf.train.Saver({"weights": theta})
```

# 使用TensorBoard


```python
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
```


```python
n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
```


```python
mse_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```


```python
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
```


```python
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index% 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
```


```python
file_writer.close()
```


```python
best_theta
```




    array([[ 2.070016  ],
           [ 0.8204561 ],
           [ 0.1173173 ],
           [-0.22739051],
           [ 0.3113402 ],
           [ 0.00353193],
           [-0.01126994],
           [-0.91643935],
           [-0.8795008 ]], dtype=float32)



# 名称作用域


```python
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
```


```python
print(error.op.name)
```

    loss/sub



```python
print(mse.op.name)
```

    loss/mse


# 模块性


```python
tf.reset_default_graph()
```


```python
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0, name="relu1")
relu2 = tf.maximum(z2, 0, name="relu2")

output = tf.add(relu1, relu2, name="output")
```


```python
#Don't Repeat Yourself
tf.reset_default_graph()
def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0, name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```


```python
file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())
```


```python
tf.reset_default_graph()
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0, name="max")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
```

# 共享变量


```python
tf.reset_default_graph()
def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu_threshold", tf.get_default_graph())
```


```python
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
```


```python
with tf.variable_scope("relu", reuse=True):#重复使用
    threshold = tf.get_variable("threshold")
```


```python
with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")
```


```python
tf.reset_default_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))

relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
```


```python
tf.reset_default_graph()

def relu(X):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer())
    w_shape = int(X.get_shape()[1]), 1
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name='bias')
    z = tf.add(tf.matmul(X, w), b, name='z')
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu7", tf.get_default_graph())
```
****
[源文件](https://github.com/coldJune/machineLearning/blob/master/handson-ml/Up%20and%20Runing%20with%20TensorFlow.ipynb)
