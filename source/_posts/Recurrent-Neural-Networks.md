---
title: Recurrent Neural Networks
date: 2018-12-28 15:43:28
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

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
```

# RNN基础

## 手动实现RNN


```python
reset_graph()

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()
```


```python
X0_batch = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 0, 1]
])# t=0
X1_batch = np.array([
    [9, 8, 7],
    [0, 0, 0],
    [6, 5, 4],
    [3, 2, 1]
])# t=1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
```


```python
print(Y0_val)
```

    [[-0.0664006   0.9625767   0.68105793  0.7091854  -0.898216  ]
     [ 0.9977755  -0.719789   -0.9965761   0.9673924  -0.9998972 ]
     [ 0.99999774 -0.99898803 -0.9999989   0.9967762  -0.9999999 ]
     [ 1.         -1.         -1.         -0.99818915  0.9995087 ]]



```python
print(Y1_val)
```

    [[ 1.         -1.         -1.          0.4020025  -0.9999998 ]
     [-0.12210419  0.62805265  0.9671843  -0.9937122  -0.2583937 ]
     [ 0.9999983  -0.9999994  -0.9999975  -0.85943305 -0.9999881 ]
     [ 0.99928284 -0.99999815 -0.9999058   0.9857963  -0.92205757]]


## 使用static_rnn()


```python
n_inputs = 3
n_neurons = 5
```


```python
reset_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
out_put_seqs, states = tf.nn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0, Y1 = out_put_seqs
```

    WARNING:tensorflow:From <ipython-input-7-64acfd881dc3>:6: BasicRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0.



```python
init = tf.global_variables_initializer()
```


```python
X0_batch = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 0, 1]
])# t=0
X1_batch = np.array([
    [9, 8, 7],
    [0, 0, 0],
    [6, 5, 4],
    [3, 2, 1]
])# t=1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
```


```python
Y0_val
```




    array([[ 0.30741334, -0.32884315, -0.6542847 , -0.9385059 ,  0.52089024],
           [ 0.99122757, -0.9542541 , -0.7518079 , -0.9995208 ,  0.9820235 ],
           [ 0.9999268 , -0.99783254, -0.8247353 , -0.9999963 ,  0.99947774],
           [ 0.996771  , -0.68750614,  0.8419969 ,  0.9303911 ,  0.8120684 ]],
          dtype=float32)




```python
Y1_val
```




    array([[ 0.99998885, -0.99976057, -0.0667929 , -0.9999803 ,  0.99982214],
           [-0.6524943 , -0.51520866, -0.37968948, -0.5922594 , -0.08968379],
           [ 0.99862397, -0.99715203, -0.03308626, -0.9991566 ,  0.9932902 ],
           [ 0.99681675, -0.9598194 ,  0.39660627, -0.8307606 ,  0.79671973]],
          dtype=float32)



## 打包序列


```python
n_steps = 2
n_inputs = 3
n_neurons = 5
```


```python
reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
out_put_seqs, states = tf.nn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(out_put_seqs), perm=[1, 0, 2])
```


```python
init = tf.global_variables_initializer()
```


```python
X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]],
])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})
```


```python
print(outputs_val)
```

    [[[-0.45652324 -0.68064123  0.40938237  0.63104504 -0.45732826]
      [-0.9428799  -0.9998869   0.94055814  0.9999985  -0.9999997 ]]

     [[-0.8001535  -0.9921827   0.7817797   0.9971032  -0.9964609 ]
      [-0.637116    0.11300927  0.5798437   0.4310559  -0.6371699 ]]

     [[-0.93605185 -0.9998379   0.9308867   0.9999815  -0.99998295]
      [-0.9165386  -0.9945604   0.896054    0.99987197 -0.9999751 ]]

     [[ 0.9927369  -0.9981933  -0.55543643  0.9989031  -0.9953323 ]
      [-0.02746338 -0.73191994  0.7827872   0.9525682  -0.9781773 ]]]



```python
print(np.transpose(outputs_val, axes=[1, 0, 2])[1])
```

    [[-0.9428799  -0.9998869   0.94055814  0.9999985  -0.9999997 ]
     [-0.637116    0.11300927  0.5798437   0.4310559  -0.6371699 ]
     [-0.9165386  -0.9945604   0.896054    0.99987197 -0.9999751 ]
     [-0.02746338 -0.73191994  0.7827872   0.9525682  -0.9781773 ]]


## 使用dynamic_rnn()


```python
n_steps = 2
n_inputs = 3
n_neurons = 5
```


```python
reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
```


```python
init = tf.global_variables_initializer()
```


```python
X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]],
])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})
```


```python
print(outputs_val)
```

    [[[-0.85115266  0.87358344  0.5802911   0.8954789  -0.0557505 ]
      [-0.999996    0.99999577  0.9981815   1.          0.37679607]]

     [[-0.9983293   0.9992038   0.98071456  0.999985    0.25192663]
      [-0.7081804  -0.0772338  -0.85227895  0.5845349  -0.78780943]]

     [[-0.9999827   0.99999535  0.9992863   1.          0.5159072 ]
      [-0.9993956   0.9984095   0.83422637  0.99999976 -0.47325212]]

     [[ 0.87888587  0.07356028  0.97216916  0.9998546  -0.7351168 ]
      [-0.9134514   0.3600957   0.7624866   0.99817705  0.80142   ]]]


## 设置序列长度


```python
n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
```


```python
seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
```


```python
init = tf.global_variables_initializer()
```


```python
X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]],
])
seq_length_batch = np.array([2, 1, 2, 2])
```


```python
with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch}
    )
```


```python
print(outputs_val)
```

    [[[-0.9123188   0.16516446  0.5548655  -0.39159346  0.20846416]
      [-1.          0.956726    0.99831694  0.99970174  0.96518576]]

     [[-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]
      [ 0.          0.          0.          0.          0.        ]]

     [[-0.99999976  0.8967997   0.9986295   0.9647514   0.93662   ]
      [-0.9999526   0.9681953   0.96002865  0.98706263  0.85459226]]

     [[-0.96435434  0.99501586 -0.36150697  0.9983378   0.999497  ]
      [-0.9613586   0.9568762   0.7132288   0.97729224 -0.0958299 ]]]



```python
print(states_val)
```

    [[-1.          0.956726    0.99831694  0.99970174  0.96518576]
     [-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]
     [-0.9999526   0.9681953   0.96002865  0.98706263  0.85459226]
     [-0.9613586   0.9568762   0.7132288   0.97729224 -0.0958299 ]]


## 训练序列分类器


```python
reset_graph()

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
```


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
X_test = X_test.reshape((-1, n_steps, n_inputs))
```


```python
n_epochs = 30
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape(-1, n_steps, n_inputs)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
```

    0 Last batch accuracy: 0.9533333 Test accuracy: 0.9288
    1 Last batch accuracy: 0.96 Test accuracy: 0.9471
    2 Last batch accuracy: 0.96 Test accuracy: 0.9499
    3 Last batch accuracy: 0.96 Test accuracy: 0.9563
    4 Last batch accuracy: 0.98 Test accuracy: 0.9677
    5 Last batch accuracy: 0.93333334 Test accuracy: 0.9651
    6 Last batch accuracy: 0.98 Test accuracy: 0.9685
    7 Last batch accuracy: 0.96666664 Test accuracy: 0.9678
    8 Last batch accuracy: 0.97333336 Test accuracy: 0.9693
    9 Last batch accuracy: 0.99333334 Test accuracy: 0.9714
    10 Last batch accuracy: 0.98 Test accuracy: 0.9752
    11 Last batch accuracy: 0.9866667 Test accuracy: 0.9743
    12 Last batch accuracy: 0.94666666 Test accuracy: 0.9716
    13 Last batch accuracy: 0.97333336 Test accuracy: 0.9658
    14 Last batch accuracy: 1.0 Test accuracy: 0.9772
    15 Last batch accuracy: 0.98 Test accuracy: 0.974
    16 Last batch accuracy: 0.99333334 Test accuracy: 0.9779
    17 Last batch accuracy: 0.9866667 Test accuracy: 0.9775
    18 Last batch accuracy: 0.9866667 Test accuracy: 0.9713
    19 Last batch accuracy: 0.98 Test accuracy: 0.9724
    20 Last batch accuracy: 0.9866667 Test accuracy: 0.9702
    21 Last batch accuracy: 0.98 Test accuracy: 0.9758
    22 Last batch accuracy: 0.98 Test accuracy: 0.9782
    23 Last batch accuracy: 0.99333334 Test accuracy: 0.9778
    24 Last batch accuracy: 0.9866667 Test accuracy: 0.9745
    25 Last batch accuracy: 0.9866667 Test accuracy: 0.9741
    26 Last batch accuracy: 0.9866667 Test accuracy: 0.9784
    27 Last batch accuracy: 1.0 Test accuracy: 0.9808
    28 Last batch accuracy: 0.99333334 Test accuracy: 0.9787
    29 Last batch accuracy: 0.98 Test accuracy: 0.9813


## 多层RNN


```python
reset_graph()
n_steps = 28
n_inputs = 28
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
```


```python
n_neurons = 100
n_layers = 3

layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,
                                     activation=tf.nn.relu)
         for layer in range(n_layers)]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
```


```python
states_concat = tf.concat(axis=1, values=states)
logits = tf.layers.dense(states_concat, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
```


```python
n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
file_writer = tf.summary.FileWriter('rnn/multiRNN', tf.get_default_graph())
```

    0 Last batch accuracy: 0.94 Test accuracy: 0.9318
    1 Last batch accuracy: 0.96 Test accuracy: 0.9563
    2 Last batch accuracy: 0.96 Test accuracy: 0.9709
    3 Last batch accuracy: 0.9866667 Test accuracy: 0.9701
    4 Last batch accuracy: 0.9866667 Test accuracy: 0.9773
    5 Last batch accuracy: 0.9533333 Test accuracy: 0.9747
    6 Last batch accuracy: 0.99333334 Test accuracy: 0.9763
    7 Last batch accuracy: 0.9866667 Test accuracy: 0.9804
    8 Last batch accuracy: 0.98 Test accuracy: 0.9755
    9 Last batch accuracy: 0.96666664 Test accuracy: 0.9804


## 时间序列


```python
t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0, n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
```


```python
t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$ t. \sin(t)/3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]),
                                      "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_49_0.png)



```python
X_batch, y_batch = next_batch(1, n_steps)
```


```python
np.c_[X_batch[0], y_batch[0]]
```




    array([[ 1.38452097,  2.05081182],
           [ 2.05081182,  2.29742291],
           [ 2.29742291,  2.0465599 ],
           [ 2.0465599 ,  1.34009916],
           [ 1.34009916,  0.32948704],
           [ 0.32948704, -0.76115235],
           [-0.76115235, -1.68967022],
           [-1.68967022, -2.25492776],
           [-2.25492776, -2.34576159],
           [-2.34576159, -1.96789418],
           [-1.96789418, -1.24220428],
           [-1.24220428, -0.37478448],
           [-0.37478448,  0.39387907],
           [ 0.39387907,  0.84815766],
           [ 0.84815766,  0.85045064],
           [ 0.85045064,  0.3752526 ],
           [ 0.3752526 , -0.48422846],
           [-0.48422846, -1.53852738],
           [-1.53852738, -2.54795941],
           [-2.54795941, -3.28097239]])



### 使用OutputProjectionWrapper


```python
reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
```


```python
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs
)
```


```python
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```


```python
learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs -y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
```


```python
saver = tf.train.Saver()
```


```python
n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    saver.save(sess, "rnn/my_time_series_model")

```

    0 	MSE: 11.967254
    100 	MSE: 0.525841
    200 	MSE: 0.1495599
    300 	MSE: 0.07279411
    400 	MSE: 0.06158535
    500 	MSE: 0.05938873
    600 	MSE: 0.05470166
    700 	MSE: 0.047849063
    800 	MSE: 0.05107608
    900 	MSE: 0.047209196
    1000 	MSE: 0.047058314
    1100 	MSE: 0.047831465
    1200 	MSE: 0.04083041
    1300 	MSE: 0.047086805
    1400 	MSE: 0.041784383



```python
with tf.Session() as sess:
    saver.restore(sess, "rnn/my_time_series_model")

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
```

    INFO:tensorflow:Restoring parameters from rnn/my_time_series_model



```python
y_pred
```




    array([[[-3.407753 ],
            [-2.4575484],
            [-1.1029298],
            [ 0.7815629],
            [ 2.2002175],
            [ 3.126768 ],
            [ 3.4037762],
            [ 3.3489153],
            [ 2.8798013],
            [ 2.2659323],
            [ 1.6447463],
            [ 1.5210768],
            [ 1.8972012],
            [ 2.7159088],
            [ 3.8894904],
            [ 5.140914 ],
            [ 6.142068 ],
            [ 6.666671 ],
            [ 6.6410103],
            [ 6.0725527]]], dtype=float32)




```python
plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]),
         "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]),
        "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0, :, 0], "r.",
        markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_61_0.png)


### 不使用OutputProjectionWrapper()


```python
reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
```


```python
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
```


```python
n_outputs = 1
learning_rate = 0.001
```


```python
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
```


```python
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    X_new  = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    saver.save(sess, "rnn/my_time_series_model")
```

    0 	MSE: 13.907029
    100 	MSE: 0.5056698
    200 	MSE: 0.19735886
    300 	MSE: 0.101214476
    400 	MSE: 0.06850145
    500 	MSE: 0.06291986
    600 	MSE: 0.055129297
    700 	MSE: 0.049436502
    800 	MSE: 0.050434686
    900 	MSE: 0.0482007
    1000 	MSE: 0.04809868
    1100 	MSE: 0.04982501
    1200 	MSE: 0.041912545
    1300 	MSE: 0.049292978
    1400 	MSE: 0.043140374



```python
y_pred
```




    array([[[-3.4332483],
            [-2.4594698],
            [-1.1081185],
            [ 0.6882153],
            [ 2.1105688],
            [ 3.0585155],
            [ 3.5144088],
            [ 3.3531117],
            [ 2.808016 ],
            [ 2.1606152],
            [ 1.662645 ],
            [ 1.5578941],
            [ 1.9173537],
            [ 2.7210245],
            [ 3.8667865],
            [ 5.100083 ],
            [ 6.099999 ],
            [ 6.6480975],
            [ 6.6147423],
            [ 6.022089 ]]], dtype=float32)




```python
plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]),
         "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]),
        "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0, :, 0], "r.",
        markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_70_0.png)


## 生成一个创造性的序列


```python
with tf.Session() as sess:
    saver.restore(sess, "rnn/my_time_series_model")

    sequence = [0.] * n_steps
    for iteration in range(300):
        X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence.append(y_pred[0, -1, 0])
```

    INFO:tensorflow:Restoring parameters from rnn/my_time_series_model



```python
plt.figure(figsize=(8, 4))
plt.plot(np.arange(len(sequence)), sequence, "b-")
plt.plot(t[:n_steps], sequence[:n_steps], "b--", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_73_0.png)



```python
with tf.Session() as sess:
    saver.restore(sess, "rnn/my_time_series_model")

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max - t_min/3))
                for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])
```

    INFO:tensorflow:Restoring parameters from rnn/my_time_series_model



```python
plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.show()
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_75_0.png)


# Deep RNN

## MultiRNNCell


```python
reset_graph()

n_inputs = 2
n_steps = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
```


```python
n_neurons = 100
n_layers = 3

layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
         for layer in range(n_layers)]

multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
```


```python
init = tf.global_variables_initializer()
```


```python
X_batch = np.random.rand(2, n_steps, n_inputs)
```


```python
with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch})
```


```python
outputs_val.shape
```




    (2, 5, 100)



## Dropout


```python
reset_graph()

n_inputs = 1
n_neurons = 100
n_layers = 3
n_steps = 20
n_outputs = 1
```


```python
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
```


```python
keep_prob = tf.placeholder_with_default(1.0, shape=())
cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
        for layer in range(n_layers)]
cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
             for cell in cells]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
```


```python
learning_rate = 0.01
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_iterations = 1500
batch_size = 50
train_keep_prob = 0.5

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        _, mse = sess.run([training_op, loss],
                          feed_dict={
                              X: X_batch,
                              y: y_batch,
                              keep_prob: train_keep_prob
                          })
        if iteration % 100 == 0:
            print(iteration, "Training MSE:", mse)
    saver.save(sess,"rnn/my_dropout_time_series_model")
```

    0 Training MSE: 16.10992
    100 Training MSE: 4.2036242
    200 Training MSE: 3.7243023
    300 Training MSE: 3.8051453
    400 Training MSE: 3.1154072
    500 Training MSE: 3.4736195
    600 Training MSE: 3.4444861
    700 Training MSE: 3.3598778
    800 Training MSE: 4.1624136
    900 Training MSE: 4.263299
    1000 Training MSE: 3.5078833
    1100 Training MSE: 4.2051315
    1200 Training MSE: 2.7443748
    1300 Training MSE: 4.583499
    1400 Training MSE: 5.121917



```python
with tf.Session() as sess:
    saver.restore(sess, "rnn/my_dropout_time_series_model")

    X_new  = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
```

    INFO:tensorflow:Restoring parameters from rnn/my_dropout_time_series_model



```python
plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]),
         "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]),
        "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0, :, 0], "r.",
        markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_91_0.png)


# LSTM


```python
reset_graph()

lstm_cell = tf.nn.rnn_cell.LSTMCell(name="basic_lstm_cell",
                                   num_units=n_neurons)
```


```python
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, name="basic_lstm_cell")
             for layer in range(n_layers)]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
```


```python
states
```




    (LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(?, 150) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(?, 150) dtype=float32>),
     LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_5:0' shape=(?, 150) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_6:0' shape=(?, 150) dtype=float32>),
     LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_7:0' shape=(?, 150) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_8:0' shape=(?, 150) dtype=float32>))




```python
top_layer_h_state
```




    <tf.Tensor 'rnn/while/Exit_8:0' shape=(?, 150) dtype=float32>




```python
n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
```

    0 Last batch accuracy: 0.9533333 Test accuracy: 0.9481
    1 Last batch accuracy: 0.96 Test accuracy: 0.9699
    2 Last batch accuracy: 0.96 Test accuracy: 0.9639
    3 Last batch accuracy: 1.0 Test accuracy: 0.9808
    4 Last batch accuracy: 0.9866667 Test accuracy: 0.9826
    5 Last batch accuracy: 1.0 Test accuracy: 0.986
    6 Last batch accuracy: 1.0 Test accuracy: 0.9872
    7 Last batch accuracy: 0.99333334 Test accuracy: 0.9882
    8 Last batch accuracy: 1.0 Test accuracy: 0.9837
    9 Last batch accuracy: 0.99333334 Test accuracy: 0.9883



```python
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, use_peepholes=True)
```


```python
gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
```

# 嵌入向量

## 获取数据


```python
from six.moves import urllib

import errno
import os
import zipfile

WORDS_PATH = "datasets/words"
WORDS_URL = "http://mattmahoney.net/dc/text8.zip"

def mkdir_p(path):
    try:
        os.makedirs(path=path)
    except OSError as exc:
        if esc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def fetch_words_data(words_rul=WORDS_URL, words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, "words.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(words_rul, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()
```


```python
words = fetch_words_data()
```


```python
words[:5]
```




    ['anarchism', 'originated', 'as', 'a', 'term']



## 建立字典


```python
from collections import Counter

vocabulary_size = 50000

vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])
```


```python
" ".join(words[:9]), data[:9]
```




    ('anarchism originated as a term of abuse first used',
     array([5234, 3081,   12,    6,  195,    2, 3134,   46,   59]))




```python
" ".join(vocabulary[word_index]
         for word_index in [5234, 3081,   12,    6,  195,    2, 3134,   46,   59])
```




    'anarchism originated as a term of abuse first used'




```python
words[24], data[24]
```




    ('culottes', 0)



## Generate batches


```python
from collections import deque

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = np.random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
```


```python
np.random.seed(42)
```


```python
data_index = 0
batch, labels = generate_batch(8, 2, 1)
```


```python
batch, [vocabulary[word] for word in batch]
```




    (array([3081, 3081,   12,   12,    6,    6,  195,  195]),
     ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term'])




```python
labels, [vocabulary[word] for word in labels[:, 0]]
```




    (array([[  12],
            [5234],
            [   6],
            [3081],
            [  12],
            [ 195],
            [   2],
            [   6]]),
     ['as', 'anarchism', 'a', 'originated', 'as', 'term', 'of', 'a'])



## 创建模型


```python
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

learning_rate = 0.01
```


```python
reset_graph()

train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
```


```python
vocabulary_size = 50000
embedding_size = 150

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)
```


```python
train_inputs = tf.placeholder(tf.int32, shape=[None])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```


```python
nce_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                       stddev= 1.0 / np.sqrt(embedding_size))
)
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```


```python
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                  num_sampled, vocabulary_size)
)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), axis=1, keepdims=True))
normalized_embedding = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embedding, transpose_b=True)

init = tf.global_variables_initializer()
```

## 训练模型


```python
num_steps = 10001

with tf.Session() as sess:
    init.run()

    average_loss = 0
    for step in range(num_steps):
        print("\rIteration:{}".format(step), end="")
        batch_inputs, batch_labels = generate_batch(batch_size,num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = sess.run([training_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ":", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocabulary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k+1]
                log_str = "Nearest to %s" % valid_word
                for k in range(top_k):
                    close_word = vocabulary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        final_embeddings = normalized_embedding.eval()
```

    Iteration:0Average loss at step 0 : 290.5275573730469
    Nearest to over tt, tuned, manichaeans, fractional, cambridge, balaguer, fluoride, strenuously,
    Nearest to one imagines, tijuana, hindrance, steadfastly, motorcyclist, lords, letting, adolfo,
    Nearest to were bezier, antibodies, nicknamed, panthers, compiler, tao, smarter, busy,
    Nearest to may failure, rna, efficacious, aspirin, lecompton, definitive, geese, amphibious,
    Nearest to two annihilate, bettors, wir, cindy, epinephrine, team, voluntarily, crystallize,
    Nearest to its knob, abeokuta, bracelet, bastards, ivens, objectivity, blanton, cold,
    Nearest to than lame, watts, stones, sram, elves, zarqawi, applets, cloves,
    Nearest to these pedro, condoned, neck, ssn, supervising, doug, thereto, melton,
    Nearest to they lowly, deportation, shrewd, reznor, tojo, decadent, occured, risotto,
    Nearest to is interests, golfers, dropouts, richards, egyptians, legionnaires, leonel, opener,
    Nearest to up clair, drives, steadfast, missed, nashville, kilowatts, anal, vinland,
    Nearest to he transitioned, winchell, resh, goldsmiths, standardised, markings, pursued, satirized,
    Nearest to people blissymbolics, mike, buffers, untouchables, carolingian, posted, ville, hypertalk,
    Nearest to more cactus, sta, reformation, poets, diligently, rsc, ravaged, nabokov,
    Nearest to was russo, rammed, investiture, glucagon, heck, adventurer, sharada, homing,
    Nearest to UNK reykjav, fi, rosalyn, mainline, archaeologist, armstrong, stevenage, ean,
    Iteration:2000Average loss at step 2000 : 133.45819056224823
    Iteration:4000Average loss at step 4000 : 62.97674214935303
    Iteration:6000Average loss at step 6000 : 40.385357957839965
    Iteration:8000Average loss at step 8000 : 31.5875605533123
    Iteration:10000Average loss at step 10000 : 25.615500225067137
    Nearest to over tikal, seal, scriptores, felony, bougainville, chapter, dubrovnik, valdemar,
    Nearest to one eight, nine, six, two, seven, four, three, five,
    Nearest to were was, logan, antlia, anaximenes, songs, by, aga, hood,
    Nearest to may zero, to, theism, eight, can, packing, would, creativity,
    Nearest to two zero, one, five, four, six, three, eight, nine,
    Nearest to its the, mechanisms, antipope, alcmene, alemanni, alexandra, alder, topped,
    Nearest to than lit, but, quantity, barbados, asmara, proxima, constructing, floors,
    Nearest to these nur, antipopes, floors, nightclubs, mainly, and, other, aurelianus,
    Nearest to they that, cain, angilbert, nine, autoerotic, alexandrovich, three, some,
    Nearest to is yahya, are, tt, but, was, stamp, ttt, politican,
    Nearest to up refrigerant, incompleteness, rensselaer, four, persistence, astor, lumped, assigned,
    Nearest to he campylobacter, his, but, eight, later, antigens, UNK, in,
    Nearest to people autoerotic, stained, trigonometry, satrap, rijndael, equality, fiesta, songs,
    Nearest to more less, ahmad, ski, conjectures, zyklon, physically, rarely, ah,
    Nearest to was became, calcite, is, were, had, asphyxiation, suffixes, nmt,
    Nearest to UNK and, one, the, dmt, a, bromide, ananda, tile,



```python
np.save("rnn/my_final_embeddings.npy", final_embeddings)
```

## plot the embeddings


```python
def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels) , "More labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
```


```python
from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [vocabulary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
```


![png](Recurrent-Neural-Networks/Recurrent%20Neural%20Networks_128_0.png)


## Machine Translation


```python
import tensorflow as tf
reset_graph()

n_steps = 50
n_neurons = 200
n_layers = 3
num_encoder_symbols = 20000
num_decoder_symbols = 20000
embedding_size = 150
learning_rate = 0.01

X = tf.placeholder(tf.int32, [None, n_steps])
Y = tf.placeholder(tf.int32, [None, n_steps])
W = tf.placeholder(tf.float32, [None, n_steps - 1, 1])
Y_input = Y[:, :-1]
Y_target = Y[:, 1:]

encoder_inputs = tf.unstack(tf.transpose(X))
decoder_inputs = tf.unstack(tf.transpose(Y_input))

lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, name="basic_lstm_cell")
            for layer in range(n_layers)]
cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

outputs_seqs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size
)

logits = tf.transpose(tf.unstack(outputs_seqs), perm=[1, 0, 2])
```


```python
logits_flat = tf.reshape(logits, [-1, num_decoder_symbols])
Y_target_flat = tf.reshape(Y_target, [-1])
W_flat = tf.reshape(W, [-1])
xentropy = W_flat * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_target_flat,
                                                                   logits=logits_flat)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
traninig_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
```
****
[源文件](https://github.com/coldJune/machineLearning/blob/master/handson-ml/Recurrent%20Neural%20Networks.ipynb)
