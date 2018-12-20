---
title: Training Deep Neural Nets
date: 2018-12-20 11:18:36
categories: 机器学习
copyright: True
tags:
    - tensorflow
    - 神经网络
    - Hands-On Machine Learning with Scikit-Learn and TensorFlow
---

# 准备


```python
import numpy as np
import tensorflow as tf

%matplotlib inline

import matplotlib.pyplot as plt


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
```

# Vanishing/Exploding Gradients Problems


```python
def logit(z):
    return 1 / (1 + np.exp(-z))
```


```python
z = np.linspace(-5, 5, 200)

plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')

plt.plot(z, logit(z), 'b--', linewidth=2)

props = dict(facecolor='black', shrink=0.1)

plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1),
             arrowprops=props, fontsize=14, ha='center')
plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0),
             arrowprops=props, fontsize=14, ha='center')
plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5),
             arrowprops=props, fontsize=14, ha='center')

plt.grid(True)
plt.title("Sigmoid activation function", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])
plt.show()
```


![png](Training-Deep-Neural-Nets/Training%20Deep%20Neural%20Nets_4_0.png)


## Xavier和 He 初始化


```python
import tensorflow as tf
reset_graph()
```


```python
n_inputs = 28 * 28
n_hidden1 = 300

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
```


```python
he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                         kernel_initializer=he_init, name="hidden1")
```

## 不饱和激活函数

### Leaky ReLU


```python
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)
```


```python
plt.plot(z, leaky_relu(z,0.05), 'b-', linewidth=2)

plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [-0.5, 4.2], 'k-')

plt.annotate('Leak', xytext=(-4.5, 0.5), xy=(-5, -0.2), arrowprops=props, ha="center")
plt.title("Leaky ReLU activation function", fontsize=14)
plt.axis([-5, 5, -0.5, 4.2])
plt.show()
```


![png](Training-Deep-Neural-Nets/Training%20Deep%20Neural%20Nets_12_0.png)



```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
```


```python
def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
```

* 使用Leaky ReLU训练MNIST


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
```


```python
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
```


```python
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
```


```python
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
```


```python
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
```


```python
with  tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accurancy = tf.reduce_mean(tf.cast(correct, tf.float32))
```


```python
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```

* 加载数据


```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
```


```python
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
```


```python
n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 5 == 0:
            acc_batch = accurancy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accurancy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accurancy:", acc_valid)
    save_path = saver.save(sess, "dnn/leaky_relu_model_final.ckpt")
file_write = tf.summary.FileWriter("dnn/leaky_relu", tf.get_default_graph())
```

    0 Batch accuracy: 0.86 Validation accurancy: 0.9044
    5 Batch accuracy: 0.94 Validation accurancy: 0.9494
    10 Batch accuracy: 0.92 Validation accurancy: 0.9656
    15 Batch accuracy: 0.94 Validation accurancy: 0.971
    20 Batch accuracy: 1.0 Validation accurancy: 0.9762
    25 Batch accuracy: 1.0 Validation accurancy: 0.9772
    30 Batch accuracy: 0.98 Validation accurancy: 0.9782
    35 Batch accuracy: 1.0 Validation accurancy: 0.9788


### ELU


```python
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z)-1), z)
```


```python
plt.plot(z, elu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], "k-")
plt.plot([0, 0], [-2.2, 3.2], "k-")
plt.plot([-5, 5], [-1, -1], "k--")
plt.title(r"ELU activation funtion($\alpha=1$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])
plt.grid(True)
plt.show()
```


![png](Training-Deep-Neural-Nets/Training%20Deep%20Neural%20Nets_29_0.png)



```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")
```

## 批量标准化(Batch Normalization)


```python
reset_graph()

import tensorflow as tf
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training = tf.placeholder_with_default(False, shape=(), name="training")

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)

```


```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training = tf.placeholder_with_default(False, shape=(), name="training")
```


```python
from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization,
                             training=training, momentum=0.9)

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = my_batch_norm_layer(hidden1)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = my_batch_norm_layer(hidden2)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = my_batch_norm_layer(logits_before_bn)
```


```python
reset_graph()

batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()

    my_batch_norm_layer = partial(
        tf.layers.batch_normalization,
        training=training,
        momentum=batch_norm_momentum
    )

    my_dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init
    )

    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))

    hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))

    logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epochs = 20
batch_size = 200
```


```python
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess, "dnn/elu_model_final.ckpt")
    file_writer = tf.summary.FileWriter("dnn/elu", tf.get_default_graph())
```

    0 Validation accuracy: 0.8952
    1 Validation accuracy: 0.9202
    2 Validation accuracy: 0.9318
    3 Validation accuracy: 0.9422
    4 Validation accuracy: 0.9468
    5 Validation accuracy: 0.954
    6 Validation accuracy: 0.9568
    7 Validation accuracy: 0.96
    8 Validation accuracy: 0.962
    9 Validation accuracy: 0.9638
    10 Validation accuracy: 0.9662
    11 Validation accuracy: 0.9682
    12 Validation accuracy: 0.9672
    13 Validation accuracy: 0.9696
    14 Validation accuracy: 0.9706
    15 Validation accuracy: 0.9704
    16 Validation accuracy: 0.9718
    17 Validation accuracy: 0.9726
    18 Validation accuracy: 0.9738
    19 Validation accuracy: 0.9742



```python
[v.name for v in tf.trainable_variables()]
```




    ['hidden1/kernel:0',
     'hidden1/bias:0',
     'batch_normalization/gamma:0',
     'batch_normalization/beta:0',
     'hidden2/kernel:0',
     'hidden2/bias:0',
     'batch_normalization_1/gamma:0',
     'batch_normalization_1/beta:0',
     'outputs/kernel:0',
     'outputs/bias:0',
     'batch_normalization_2/gamma:0',
     'batch_normalization_2/beta:0']




```python
[v.name for v in tf.global_variables()]
```




    ['hidden1/kernel:0',
     'hidden1/bias:0',
     'batch_normalization/gamma:0',
     'batch_normalization/beta:0',
     'batch_normalization/moving_mean:0',
     'batch_normalization/moving_variance:0',
     'hidden2/kernel:0',
     'hidden2/bias:0',
     'batch_normalization_1/gamma:0',
     'batch_normalization_1/beta:0',
     'batch_normalization_1/moving_mean:0',
     'batch_normalization_1/moving_variance:0',
     'outputs/kernel:0',
     'outputs/bias:0',
     'batch_normalization_2/gamma:0',
     'batch_normalization_2/beta:0',
     'batch_normalization_2/moving_mean:0',
     'batch_normalization_2/moving_variance:0']



## 梯度裁剪


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
```


```python
learning_rate = 0.01
```


```python
threhold = 1.0

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threhold, threhold), var)
             for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)
```


```python
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
```


```python
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epochs = 20
batch_size = 200
```


```python
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "dnn/clip_model_final.ckpt")
filewriter = tf.summary.FileWriter("dnn/clip", tf.get_default_graph())
```

    0 Validation accuracy: 0.2906
    1 Validation accuracy: 0.795
    2 Validation accuracy: 0.8836
    3 Validation accuracy: 0.9068
    4 Validation accuracy: 0.9136
    5 Validation accuracy: 0.9232
    6 Validation accuracy: 0.93
    7 Validation accuracy: 0.9342
    8 Validation accuracy: 0.9384
    9 Validation accuracy: 0.944
    10 Validation accuracy: 0.9454
    11 Validation accuracy: 0.9472
    12 Validation accuracy: 0.9516
    13 Validation accuracy: 0.9532
    14 Validation accuracy: 0.9542
    15 Validation accuracy: 0.9562
    16 Validation accuracy: 0.9572
    17 Validation accuracy: 0.9596
    18 Validation accuracy: 0.9588
    19 Validation accuracy: 0.9616


# 重利用之前训练的层

## 重利用TensorFlow模型


```python
reset_graph()
saver = tf.train.import_meta_graph("dnn/clip_model_final.ckpt.meta")
```


```python
for op in tf.get_default_graph().get_operations():
    print(op.name)
```

    X
    y
    hidden1/kernel/Initializer/random_uniform/shape
    hidden1/kernel/Initializer/random_uniform/min
    hidden1/kernel/Initializer/random_uniform/max
    hidden1/kernel/Initializer/random_uniform/RandomUniform
    hidden1/kernel/Initializer/random_uniform/sub
    hidden1/kernel/Initializer/random_uniform/mul
    hidden1/kernel/Initializer/random_uniform
    hidden1/kernel
    hidden1/kernel/Assign
    hidden1/kernel/read
    hidden1/bias/Initializer/zeros
    hidden1/bias
    hidden1/bias/Assign
    hidden1/bias/read
    dnn/hidden1/MatMul
    dnn/hidden1/BiasAdd
    dnn/hidden1/Relu
    hidden2/kernel/Initializer/random_uniform/shape
    hidden2/kernel/Initializer/random_uniform/min
    hidden2/kernel/Initializer/random_uniform/max
    hidden2/kernel/Initializer/random_uniform/RandomUniform
    hidden2/kernel/Initializer/random_uniform/sub
    hidden2/kernel/Initializer/random_uniform/mul
    hidden2/kernel/Initializer/random_uniform
    hidden2/kernel
    hidden2/kernel/Assign
    hidden2/kernel/read
    hidden2/bias/Initializer/zeros
    hidden2/bias
    hidden2/bias/Assign
    hidden2/bias/read
    dnn/hidden2/MatMul
    dnn/hidden2/BiasAdd
    dnn/hidden2/Relu
    hidden3/kernel/Initializer/random_uniform/shape
    hidden3/kernel/Initializer/random_uniform/min
    hidden3/kernel/Initializer/random_uniform/max
    hidden3/kernel/Initializer/random_uniform/RandomUniform
    hidden3/kernel/Initializer/random_uniform/sub
    hidden3/kernel/Initializer/random_uniform/mul
    hidden3/kernel/Initializer/random_uniform
    hidden3/kernel
    hidden3/kernel/Assign
    hidden3/kernel/read
    hidden3/bias/Initializer/zeros
    hidden3/bias
    hidden3/bias/Assign
    hidden3/bias/read
    dnn/hidden3/MatMul
    dnn/hidden3/BiasAdd
    dnn/hidden3/Relu
    hidden4/kernel/Initializer/random_uniform/shape
    hidden4/kernel/Initializer/random_uniform/min
    hidden4/kernel/Initializer/random_uniform/max
    hidden4/kernel/Initializer/random_uniform/RandomUniform
    hidden4/kernel/Initializer/random_uniform/sub
    hidden4/kernel/Initializer/random_uniform/mul
    hidden4/kernel/Initializer/random_uniform
    hidden4/kernel
    hidden4/kernel/Assign
    hidden4/kernel/read
    hidden4/bias/Initializer/zeros
    hidden4/bias
    hidden4/bias/Assign
    hidden4/bias/read
    dnn/hidden4/MatMul
    dnn/hidden4/BiasAdd
    dnn/hidden4/Relu
    hidden5/kernel/Initializer/random_uniform/shape
    hidden5/kernel/Initializer/random_uniform/min
    hidden5/kernel/Initializer/random_uniform/max
    hidden5/kernel/Initializer/random_uniform/RandomUniform
    hidden5/kernel/Initializer/random_uniform/sub
    hidden5/kernel/Initializer/random_uniform/mul
    hidden5/kernel/Initializer/random_uniform
    hidden5/kernel
    hidden5/kernel/Assign
    hidden5/kernel/read
    hidden5/bias/Initializer/zeros
    hidden5/bias
    hidden5/bias/Assign
    hidden5/bias/read
    dnn/hidden5/MatMul
    dnn/hidden5/BiasAdd
    dnn/hidden5/Relu
    outputs/kernel/Initializer/random_uniform/shape
    outputs/kernel/Initializer/random_uniform/min
    outputs/kernel/Initializer/random_uniform/max
    outputs/kernel/Initializer/random_uniform/RandomUniform
    outputs/kernel/Initializer/random_uniform/sub
    outputs/kernel/Initializer/random_uniform/mul
    outputs/kernel/Initializer/random_uniform
    outputs/kernel
    outputs/kernel/Assign
    outputs/kernel/read
    outputs/bias/Initializer/zeros
    outputs/bias
    outputs/bias/Assign
    outputs/bias/read
    dnn/outputs/MatMul
    dnn/outputs/BiasAdd
    loss/SparseSoftmaxCrossEntropyWithLogits/Shape
    loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
    loss/Const
    loss/loss
    gradients/Shape
    gradients/grad_ys_0
    gradients/Fill
    gradients/loss/loss_grad/Reshape/shape
    gradients/loss/loss_grad/Reshape
    gradients/loss/loss_grad/Shape
    gradients/loss/loss_grad/Tile
    gradients/loss/loss_grad/Shape_1
    gradients/loss/loss_grad/Shape_2
    gradients/loss/loss_grad/Const
    gradients/loss/loss_grad/Prod
    gradients/loss/loss_grad/Const_1
    gradients/loss/loss_grad/Prod_1
    gradients/loss/loss_grad/Maximum/y
    gradients/loss/loss_grad/Maximum
    gradients/loss/loss_grad/floordiv
    gradients/loss/loss_grad/Cast
    gradients/loss/loss_grad/truediv
    gradients/zeros_like
    gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient
    gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim
    gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
    gradients/loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul
    gradients/dnn/outputs/BiasAdd_grad/BiasAddGrad
    gradients/dnn/outputs/BiasAdd_grad/tuple/group_deps
    gradients/dnn/outputs/BiasAdd_grad/tuple/control_dependency
    gradients/dnn/outputs/BiasAdd_grad/tuple/control_dependency_1
    gradients/dnn/outputs/MatMul_grad/MatMul
    gradients/dnn/outputs/MatMul_grad/MatMul_1
    gradients/dnn/outputs/MatMul_grad/tuple/group_deps
    gradients/dnn/outputs/MatMul_grad/tuple/control_dependency
    gradients/dnn/outputs/MatMul_grad/tuple/control_dependency_1
    gradients/dnn/hidden5/Relu_grad/ReluGrad
    gradients/dnn/hidden5/BiasAdd_grad/BiasAddGrad
    gradients/dnn/hidden5/BiasAdd_grad/tuple/group_deps
    gradients/dnn/hidden5/BiasAdd_grad/tuple/control_dependency
    gradients/dnn/hidden5/BiasAdd_grad/tuple/control_dependency_1
    gradients/dnn/hidden5/MatMul_grad/MatMul
    gradients/dnn/hidden5/MatMul_grad/MatMul_1
    gradients/dnn/hidden5/MatMul_grad/tuple/group_deps
    gradients/dnn/hidden5/MatMul_grad/tuple/control_dependency
    gradients/dnn/hidden5/MatMul_grad/tuple/control_dependency_1
    gradients/dnn/hidden4/Relu_grad/ReluGrad
    gradients/dnn/hidden4/BiasAdd_grad/BiasAddGrad
    gradients/dnn/hidden4/BiasAdd_grad/tuple/group_deps
    gradients/dnn/hidden4/BiasAdd_grad/tuple/control_dependency
    gradients/dnn/hidden4/BiasAdd_grad/tuple/control_dependency_1
    gradients/dnn/hidden4/MatMul_grad/MatMul
    gradients/dnn/hidden4/MatMul_grad/MatMul_1
    gradients/dnn/hidden4/MatMul_grad/tuple/group_deps
    gradients/dnn/hidden4/MatMul_grad/tuple/control_dependency
    gradients/dnn/hidden4/MatMul_grad/tuple/control_dependency_1
    gradients/dnn/hidden3/Relu_grad/ReluGrad
    gradients/dnn/hidden3/BiasAdd_grad/BiasAddGrad
    gradients/dnn/hidden3/BiasAdd_grad/tuple/group_deps
    gradients/dnn/hidden3/BiasAdd_grad/tuple/control_dependency
    gradients/dnn/hidden3/BiasAdd_grad/tuple/control_dependency_1
    gradients/dnn/hidden3/MatMul_grad/MatMul
    gradients/dnn/hidden3/MatMul_grad/MatMul_1
    gradients/dnn/hidden3/MatMul_grad/tuple/group_deps
    gradients/dnn/hidden3/MatMul_grad/tuple/control_dependency
    gradients/dnn/hidden3/MatMul_grad/tuple/control_dependency_1
    gradients/dnn/hidden2/Relu_grad/ReluGrad
    gradients/dnn/hidden2/BiasAdd_grad/BiasAddGrad
    gradients/dnn/hidden2/BiasAdd_grad/tuple/group_deps
    gradients/dnn/hidden2/BiasAdd_grad/tuple/control_dependency
    gradients/dnn/hidden2/BiasAdd_grad/tuple/control_dependency_1
    gradients/dnn/hidden2/MatMul_grad/MatMul
    gradients/dnn/hidden2/MatMul_grad/MatMul_1
    gradients/dnn/hidden2/MatMul_grad/tuple/group_deps
    gradients/dnn/hidden2/MatMul_grad/tuple/control_dependency
    gradients/dnn/hidden2/MatMul_grad/tuple/control_dependency_1
    gradients/dnn/hidden1/Relu_grad/ReluGrad
    gradients/dnn/hidden1/BiasAdd_grad/BiasAddGrad
    gradients/dnn/hidden1/BiasAdd_grad/tuple/group_deps
    gradients/dnn/hidden1/BiasAdd_grad/tuple/control_dependency
    gradients/dnn/hidden1/BiasAdd_grad/tuple/control_dependency_1
    gradients/dnn/hidden1/MatMul_grad/MatMul
    gradients/dnn/hidden1/MatMul_grad/MatMul_1
    gradients/dnn/hidden1/MatMul_grad/tuple/group_deps
    gradients/dnn/hidden1/MatMul_grad/tuple/control_dependency
    gradients/dnn/hidden1/MatMul_grad/tuple/control_dependency_1
    clip_by_value/Minimum/y
    clip_by_value/Minimum
    clip_by_value/y
    clip_by_value
    clip_by_value_1/Minimum/y
    clip_by_value_1/Minimum
    clip_by_value_1/y
    clip_by_value_1
    clip_by_value_2/Minimum/y
    clip_by_value_2/Minimum
    clip_by_value_2/y
    clip_by_value_2
    clip_by_value_3/Minimum/y
    clip_by_value_3/Minimum
    clip_by_value_3/y
    clip_by_value_3
    clip_by_value_4/Minimum/y
    clip_by_value_4/Minimum
    clip_by_value_4/y
    clip_by_value_4
    clip_by_value_5/Minimum/y
    clip_by_value_5/Minimum
    clip_by_value_5/y
    clip_by_value_5
    clip_by_value_6/Minimum/y
    clip_by_value_6/Minimum
    clip_by_value_6/y
    clip_by_value_6
    clip_by_value_7/Minimum/y
    clip_by_value_7/Minimum
    clip_by_value_7/y
    clip_by_value_7
    clip_by_value_8/Minimum/y
    clip_by_value_8/Minimum
    clip_by_value_8/y
    clip_by_value_8
    clip_by_value_9/Minimum/y
    clip_by_value_9/Minimum
    clip_by_value_9/y
    clip_by_value_9
    clip_by_value_10/Minimum/y
    clip_by_value_10/Minimum
    clip_by_value_10/y
    clip_by_value_10
    clip_by_value_11/Minimum/y
    clip_by_value_11/Minimum
    clip_by_value_11/y
    clip_by_value_11
    GradientDescent/learning_rate
    GradientDescent/update_hidden1/kernel/ApplyGradientDescent
    GradientDescent/update_hidden1/bias/ApplyGradientDescent
    GradientDescent/update_hidden2/kernel/ApplyGradientDescent
    GradientDescent/update_hidden2/bias/ApplyGradientDescent
    GradientDescent/update_hidden3/kernel/ApplyGradientDescent
    GradientDescent/update_hidden3/bias/ApplyGradientDescent
    GradientDescent/update_hidden4/kernel/ApplyGradientDescent
    GradientDescent/update_hidden4/bias/ApplyGradientDescent
    GradientDescent/update_hidden5/kernel/ApplyGradientDescent
    GradientDescent/update_hidden5/bias/ApplyGradientDescent
    GradientDescent/update_outputs/kernel/ApplyGradientDescent
    GradientDescent/update_outputs/bias/ApplyGradientDescent
    GradientDescent
    eval/in_top_k/InTopKV2/k
    eval/in_top_k/InTopKV2
    eval/Cast
    eval/Const
    eval/accuracy
    init
    save/Const
    save/SaveV2/tensor_names
    save/SaveV2/shape_and_slices
    save/SaveV2
    save/control_dependency
    save/RestoreV2/tensor_names
    save/RestoreV2/shape_and_slices
    save/RestoreV2
    save/Assign
    save/Assign_1
    save/Assign_2
    save/Assign_3
    save/Assign_4
    save/Assign_5
    save/Assign_6
    save/Assign_7
    save/Assign_8
    save/Assign_9
    save/Assign_10
    save/Assign_11
    save/restore_all



```python
#选择需要的
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")

training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")
```


```python
# 将重要的操作放在单独的集合中
for op in (X, y, accuracy, training_op):
    tf.add_to_collection("my_important_ops", op)
```


```python
# 或许这些操作
X, y, accuracy, training_op = tf.get_collection("my_important_ops")
```


```python
with tf.Session() as sess:
    saver.restore(sess, "dnn/clip_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess, "dnn/my_new_model_final.ckpt")
```

    INFO:tensorflow:Restoring parameters from dnn/clip_model_final.ckpt
    0 Validation accuracy: 0.9626
    1 Validation accuracy: 0.963
    2 Validation accuracy: 0.9632
    3 Validation accuracy: 0.9658
    4 Validation accuracy: 0.965
    5 Validation accuracy: 0.9628
    6 Validation accuracy: 0.966
    7 Validation accuracy: 0.9678
    8 Validation accuracy: 0.9672
    9 Validation accuracy: 0.9678
    10 Validation accuracy: 0.97
    11 Validation accuracy: 0.97
    12 Validation accuracy: 0.966
    13 Validation accuracy: 0.9706
    14 Validation accuracy: 0.972
    15 Validation accuracy: 0.9708
    16 Validation accuracy: 0.9724
    17 Validation accuracy: 0.9706
    18 Validation accuracy: 0.972
    19 Validation accuracy: 0.9684



```python
# 重利用低层
reset_graph()
n_hidden4 = 20
n_outputs = 10

saver = tf.train.import_meta_graph("dnn/clip_model_final.ckpt.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden3/Relu:0")

new_hidden4 = tf.layers.dense(hidden3, n_hidden4,
                              activation=tf.nn.relu, name="new_hidden4")
new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="losss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
new_saver = tf.train.Saver()
```


```python
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "dnn/clip_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = new_saver.save(sess, "dnn/my_new_low_layer_model_final.ckpt")
```

    INFO:tensorflow:Restoring parameters from dnn/clip_model_final.ckpt
    0 Validation accuracy: 0.9172
    1 Validation accuracy: 0.9394
    2 Validation accuracy: 0.9464
    3 Validation accuracy: 0.95
    4 Validation accuracy: 0.955
    5 Validation accuracy: 0.9522
    6 Validation accuracy: 0.9566
    7 Validation accuracy: 0.9598
    8 Validation accuracy: 0.9608
    9 Validation accuracy: 0.9608
    10 Validation accuracy: 0.9628
    11 Validation accuracy: 0.9622
    12 Validation accuracy: 0.9646
    13 Validation accuracy: 0.9648
    14 Validation accuracy: 0.9654
    15 Validation accuracy: 0.9668
    16 Validation accuracy: 0.9676
    17 Validation accuracy: 0.9662
    18 Validation accuracy: 0.9684
    19 Validation accuracy: 0.968


## 冻结低层


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 20
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope="hidden[34]|outputs")
    training_op = optimizer.minimize(loss, var_list=train_vars)

init  = tf.global_variables_initializer()
new_saver = tf.train.Saver()


reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]")
restore_saver = tf.train.Saver(reuse_vars)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "dnn/clip_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = new_saver.save(sess, "dnn/my_new_freeze_low_layer_model_final.ckpt")
```

    INFO:tensorflow:Restoring parameters from dnn/clip_model_final.ckpt
    0 Validation accuracy: 0.893
    1 Validation accuracy: 0.9234
    2 Validation accuracy: 0.9358
    3 Validation accuracy: 0.941
    4 Validation accuracy: 0.947
    5 Validation accuracy: 0.9484
    6 Validation accuracy: 0.95
    7 Validation accuracy: 0.954
    8 Validation accuracy: 0.9542
    9 Validation accuracy: 0.954
    10 Validation accuracy: 0.955
    11 Validation accuracy: 0.9548
    12 Validation accuracy: 0.9572
    13 Validation accuracy: 0.957
    14 Validation accuracy: 0.9564
    15 Validation accuracy: 0.957
    16 Validation accuracy: 0.9576
    17 Validation accuracy: 0.958
    18 Validation accuracy: 0.9586
    19 Validation accuracy: 0.9582


## 缓冲冻结的层


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 20
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden2_stop = tf.stop_gradient(hidden2)
    hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]")
restore_saver = tf.train.Saver(reuse_vars)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
import numpy as np

n_batches = len(X_train) // batch_size

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "dnn/clip_model_final.ckpt")

    h2_cache = sess.run(hidden2, feed_dict={X: X_train})
    h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid})

    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(len(X_train))
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(y_train[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess, "dnn/my_new_cache_freeze_low_layer_model_final.ckpt")      
```

    INFO:tensorflow:Restoring parameters from dnn/clip_model_final.ckpt
    0 Validation accuracy: 0.9006
    1 Validation accuracy: 0.9346
    2 Validation accuracy: 0.9444
    3 Validation accuracy: 0.9478
    4 Validation accuracy: 0.9516
    5 Validation accuracy: 0.952
    6 Validation accuracy: 0.9522
    7 Validation accuracy: 0.9532
    8 Validation accuracy: 0.9546
    9 Validation accuracy: 0.9558
    10 Validation accuracy: 0.955
    11 Validation accuracy: 0.9554
    12 Validation accuracy: 0.9558
    13 Validation accuracy: 0.9572
    14 Validation accuracy: 0.957
    15 Validation accuracy: 0.9572
    16 Validation accuracy: 0.9568
    17 Validation accuracy: 0.9592
    18 Validation accuracy: 0.958
    19 Validation accuracy: 0.9598


# 更快的优化器

## Momentum optimization


```python
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
```

## Nesterov Accelerated Gradient


```python
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9, use_nesterov=True)
```

## AdaGrad


```python
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
```

## RMSProp


```python
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9,
                                     decay=0.9, epsilon=1e-10)
```

## Adam Optimization


```python
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
```

## Learning Rate Scheduling


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                              decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "dnn/learning_rate_scheduling_model_final.ckpt")
```

    0 Validation accuracy: 0.9574
    1 Validation accuracy: 0.9716
    2 Validation accuracy: 0.973
    3 Validation accuracy: 0.9798
    4 Validation accuracy: 0.9816


# 通过正规化避免过拟合

## $l_1$ 和 $l_2$ 正规化

 * 手动实现 $l_1$ 正规化


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

scale = 0.001

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
    loss = tf.add(base_loss, scale * reg_losses, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epoches = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess,"dnn/l_1_model_final.ckpt")
```

    0 Validation accuracy: 0.831
    1 Validation accuracy: 0.871
    2 Validation accuracy: 0.8838
    3 Validation accuracy: 0.8934
    4 Validation accuracy: 0.8966
    5 Validation accuracy: 0.8988
    6 Validation accuracy: 0.9016
    7 Validation accuracy: 0.9044
    8 Validation accuracy: 0.9058
    9 Validation accuracy: 0.906
    10 Validation accuracy: 0.9068
    11 Validation accuracy: 0.9054
    12 Validation accuracy: 0.907
    13 Validation accuracy: 0.9084
    14 Validation accuracy: 0.9088
    15 Validation accuracy: 0.9064
    16 Validation accuracy: 0.9068
    17 Validation accuracy: 0.9066
    18 Validation accuracy: 0.9066
    19 Validation accuracy: 0.9052


* 使用正规化方法


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

scale = 0.001

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
    loss = tf.add(base_loss, scale * reg_losses, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

```


```python
n_epoches = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess,"dnn/l_1_model_final.ckpt")
```

    0 Validation accuracy: 0.831
    1 Validation accuracy: 0.871
    2 Validation accuracy: 0.8838
    3 Validation accuracy: 0.8934
    4 Validation accuracy: 0.8966
    5 Validation accuracy: 0.8988
    6 Validation accuracy: 0.9016
    7 Validation accuracy: 0.9044
    8 Validation accuracy: 0.9058
    9 Validation accuracy: 0.906
    10 Validation accuracy: 0.9068
    11 Validation accuracy: 0.9054
    12 Validation accuracy: 0.907
    13 Validation accuracy: 0.9084
    14 Validation accuracy: 0.9088
    15 Validation accuracy: 0.9064
    16 Validation accuracy: 0.9068
    17 Validation accuracy: 0.9066
    18 Validation accuracy: 0.9066
    19 Validation accuracy: 0.9052



```python
reset_graph()
from functools import partial
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
scale = 0.001

my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale)
)

with tf.name_scope("dnn"):
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, activation=None, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epoches = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess,"dnn/l_1_function_model_final.ckpt")
```

    0 Validation accuracy: 0.8274
    1 Validation accuracy: 0.8766
    2 Validation accuracy: 0.8952
    3 Validation accuracy: 0.9016
    4 Validation accuracy: 0.908
    5 Validation accuracy: 0.9096
    6 Validation accuracy: 0.9124
    7 Validation accuracy: 0.9154
    8 Validation accuracy: 0.9178
    9 Validation accuracy: 0.919
    10 Validation accuracy: 0.92
    11 Validation accuracy: 0.9224
    12 Validation accuracy: 0.9212
    13 Validation accuracy: 0.9228
    14 Validation accuracy: 0.9222
    15 Validation accuracy: 0.9218
    16 Validation accuracy: 0.9218
    17 Validation accuracy: 0.9228
    18 Validation accuracy: 0.9216
    19 Validation accuracy: 0.9214


## DropOut


```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
```


```python
training = tf.placeholder_with_default(False, shape=(), name="training")

dropout_rate = 0.5
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                             name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)

    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                             name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)

    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epoches = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    save_path = saver.save(sess,"dnn/dropout_model_final.ckpt")
```

    0 Validation accuracy: 0.923
    1 Validation accuracy: 0.9438
    2 Validation accuracy: 0.9504
    3 Validation accuracy: 0.961
    4 Validation accuracy: 0.9654
    5 Validation accuracy: 0.9694
    6 Validation accuracy: 0.9726
    7 Validation accuracy: 0.9736
    8 Validation accuracy: 0.9756
    9 Validation accuracy: 0.975
    10 Validation accuracy: 0.9768
    11 Validation accuracy: 0.9782
    12 Validation accuracy: 0.976
    13 Validation accuracy: 0.9788
    14 Validation accuracy: 0.9776
    15 Validation accuracy: 0.9802
    16 Validation accuracy: 0.9796
    17 Validation accuracy: 0.9802
    18 Validation accuracy: 0.9806
    19 Validation accuracy: 0.9806


## Max Norm


```python
def max_norm_regularizer(threshold, axes=1, name="max_norm",
                        collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None
    return max_norm
```


```python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
```


```python
max_norm_reg = max_norm_regularizer(threshold=1.0)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                             kernel_regularizer=max_norm_reg, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                             kernel_regularizer=max_norm_reg, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
```


```python
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epoches = 20
batch_size = 50
```


```python
clip_all_weights = tf.get_collection("max_norm")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)
        acc_valid = accuracy.eval(feed_dict={X:X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", acc_valid)
    save_path = saver.save(sess, "dnn/max_norm_model_final.ckpt")    
```

    0 Validation accuracy: 0.9556
    1 Validation accuracy: 0.97
    2 Validation accuracy: 0.973
    3 Validation accuracy: 0.9758
    4 Validation accuracy: 0.9762
    5 Validation accuracy: 0.9788
    6 Validation accuracy: 0.98
    7 Validation accuracy: 0.9824
    8 Validation accuracy: 0.9816
    9 Validation accuracy: 0.981
    10 Validation accuracy: 0.983
    11 Validation accuracy: 0.982
    12 Validation accuracy: 0.9808
    13 Validation accuracy: 0.9828
    14 Validation accuracy: 0.982
    15 Validation accuracy: 0.9824
    16 Validation accuracy: 0.9824
    17 Validation accuracy: 0.9824
    18 Validation accuracy: 0.982
    19 Validation accuracy: 0.9824
 ****
 [源文件](https://github.com/coldJune/machineLearning/blob/master/handson-ml/Training%20Deep%20Neural%20Nets.ipynb)
