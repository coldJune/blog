---
title: Introduction to Artificial Neural Networks
date: 2018-12-17 11:13:40
categories: 机器学习
copyright: True
tags:
    - tensorflow
    - 神经网络
    - Hands-On Machine Learning with Scikit-Learn and TensorFlow
---

# 感知器


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)].astype(np.float32)
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state=42, max_iter=50, tol=1e-3)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
```


```python
y_pred
```




    array([1])




```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss="perceptron", learning_rate="constant",
                        eta0=1, penalty=None, max_iter=50, tol=1e-3)
sgd_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
```


```python
y_pred
```




    array([1])



# 使用TensorFlow高级API


```python
import tensorflow as tf

(X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

```


```python
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                        feature_columns=feature_columns)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)
```
    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\deng.xj\AppData\Local\Temp\tmpoqlzthaw
    INFO:tensorflow:Using config: {'_task_type': None, '_task_id': 0, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000001BB66D724A8>, '_master': '', '_num_ps_replicas': 0, '_num_worker_replicas': 0, '_environment': 'local', '_is_chief': True, '_evaluation_master': '', '_train_distribute': None, '_eval_distribute': None, '_device_fn': None, '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1.0
    }
    , '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_secs': 600, '_log_step_count_steps': 100, '_protocol': None, '_session_config': None, '_save_checkpoints_steps': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': 'C:\\Users\\deng.xj\\AppData\\Local\\Temp\\tmpoqlzthaw'}
    WARNING:tensorflow:From <ipython-input-6-90ea1841712a>:4: calling BaseEstimator.fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
    Instructions for updating:
    Estimator is decoupled from Scikit Learn interface by moving into
    separate class SKCompat. Arguments x, y and batch_size are only
    available in the SKCompat class, Estimator will only accept input_fn.
    Example conversion:
      est = Estimator(...) -> est = SKCompat(Estimator(...))
    WARNING:tensorflow:From <ipython-input-6-90ea1841712a>:4: calling BaseEstimator.fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with y is deprecated and will be removed after 2016-12-01.
    Instructions for updating:
    Estimator is decoupled from Scikit Learn interface by moving into
    separate class SKCompat. Arguments x, y and batch_size are only
    available in the SKCompat class, Estimator will only accept input_fn.
    Example conversion:
      est = Estimator(...) -> est = SKCompat(Estimator(...))
    WARNING:tensorflow:From <ipython-input-6-90ea1841712a>:4: calling BaseEstimator.fit (from tensorflow.contrib.learn.python.learn.estimators.estimator) with batch_size is deprecated and will be removed after 2016-12-01.
    Instructions for updating:
    Estimator is decoupled from Scikit Learn interface by moving into
    separate class SKCompat. Arguments x, y and batch_size are only
    available in the SKCompat class, Estimator will only accept input_fn.
    Example conversion:
      est = Estimator(...) -> est = SKCompat(Estimator(...))
    WARNING:tensorflow:From e:\python\python36\lib\site-packages\tensorflow\contrib\learn\python\learn\estimators\estimator.py:509: SKCompat.__init__ (from tensorflow.contrib.learn.python.learn.estimators.estimator) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please switch to the Estimator interface.
    WARNING:tensorflow:From e:\python\python36\lib\site-packages\tensorflow\contrib\learn\python\learn\learn_io\data_feeder.py:102: extract_pandas_labels (from tensorflow.contrib.learn.python.learn.learn_io.pandas_io) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please access pandas data directly.
    WARNING:tensorflow:From e:\python\python36\lib\site-packages\tensorflow\contrib\learn\python\learn\estimators\head.py:678: ModelFnOps.__new__ (from tensorflow.contrib.learn.python.learn.estimators.model_fn) is deprecated and will be removed in a future version.
    Instructions for updating:
    When switching to tf.estimator.Estimator, use tf.estimator.EstimatorSpec. You can use the `estimator_spec` method to create an equivalent one.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into C:\Users\deng.xj\AppData\Local\Temp\tmpoqlzthaw\model.ckpt.
    INFO:tensorflow:loss = 2.292883, step = 1
    INFO:tensorflow:global_step/sec: 238.165
    INFO:tensorflow:loss = 0.32534814, step = 101 (0.422 sec)
    INFO:tensorflow:global_step/sec: 338.742
    ...
    INFO:tensorflow:loss = 0.0007182892, step = 39601 (0.298 sec)
    INFO:tensorflow:global_step/sec: 333.114
    INFO:tensorflow:loss = 0.00014897775, step = 39701 (0.300 sec)
    INFO:tensorflow:global_step/sec: 342.208
    INFO:tensorflow:loss = 0.0010287359, step = 39801 (0.292 sec)
    INFO:tensorflow:global_step/sec: 333.116
    INFO:tensorflow:loss = 0.0009217943, step = 39901 (0.300 sec)
    INFO:tensorflow:Saving checkpoints for 40000 into C:\Users\deng.xj\AppData\Local\Temp\tmpoqlzthaw\model.ckpt.
    INFO:tensorflow:Loss for final step: 0.00035940565.





    DNNClassifier(params={'head': <tensorflow.contrib.learn.python.learn.estimators.head._MultiClassHead object at 0x000001BB6821B668>, 'hidden_units': [300, 100], 'feature_columns': (_RealValuedColumn(column_name='', dimension=784, default_value=None, dtype=tf.float32, normalizer=None),), 'optimizer': None, 'activation_fn': <function relu at 0x000001BB6378FF28>, 'dropout': None, 'gradient_clip_norm': None, 'embedding_lr_multipliers': None, 'input_layer_min_slice_size': None})




```python
from sklearn.metrics import accuracy_score
y_pred = list(dnn_clf.predict(X_test))
accuracy_score(y_test,y_pred)
```

    WARNING:tensorflow:From e:\python\python36\lib\site-packages\tensorflow\python\util\deprecation.py:553: calling DNNClassifier.predict (from tensorflow.contrib.learn.python.learn.estimators.dnn) with outputs=None is deprecated and will be removed after 2017-03-01.
    Instructions for updating:
    Please switch to predict_classes, or set `outputs` argument.
    WARNING:tensorflow:From e:\python\python36\lib\site-packages\tensorflow\contrib\learn\python\learn\estimators\dnn.py:463: calling BaseEstimator.predict (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
    Instructions for updating:
    Estimator is decoupled from Scikit Learn interface by moving into
    separate class SKCompat. Arguments x, y and batch_size are only
    available in the SKCompat class, Estimator will only accept input_fn.
    Example conversion:
      est = Estimator(...) -> est = SKCompat(Estimator(...))
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\deng.xj\AppData\Local\Temp\tmpoqlzthaw\model.ckpt-40000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.





    0.9814




```python
dnn_clf.evaluate(X_test, y_test)
```

    WARNING:tensorflow:From <ipython-input-8-862f84b3278e>:1: calling BaseEstimator.evaluate (from tensorflow.contrib.learn.python.learn.estimators.estimator) with x is deprecated and will be removed after 2016-12-01.
    Instructions for updating:
    Estimator is decoupled from Scikit Learn interface by moving into
    separate class SKCompat. Arguments x, y and batch_size are only
    available in the SKCompat class, Estimator will only accept input_fn.
    Example conversion:
      est = Estimator(...) -> est = SKCompat(Estimator(...))
    WARNING:tensorflow:From <ipython-input-8-862f84b3278e>:1: calling BaseEstimator.evaluate (from tensorflow.contrib.learn.python.learn.estimators.estimator) with y is deprecated and will be removed after 2016-12-01.
    Instructions for updating:
    Estimator is decoupled from Scikit Learn interface by moving into
    separate class SKCompat. Arguments x, y and batch_size are only
    available in the SKCompat class, Estimator will only accept input_fn.
    Example conversion:
      est = Estimator(...) -> est = SKCompat(Estimator(...))
    INFO:tensorflow:Starting evaluation at 2018-12-17-02:39:23
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from C:\Users\deng.xj\AppData\Local\Temp\tmpoqlzthaw\model.ckpt-40000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2018-12-17-02:39:23
    INFO:tensorflow:Saving dict for global step 40000: accuracy = 0.9814, global_step = 40000, loss = 0.07299631





    {'accuracy': 0.9814, 'global_step': 40000, 'loss': 0.07299631}



# 手写DNN

## 数据构造


```python
tf.reset_default_graph()
```


```python
import tensorflow as tf
import tensorboard as tb

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
```


```python
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
```


```python
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        # 每一层建立一个名字空间
        n_inputs = int(X.get_shape()[1])# 获取特征数
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z
```


```python
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits =neuron_layer(hidden2, n_outputs, "outputs")
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
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```


```python
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```

## 执行


```python
n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
```


```python
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
    save_path = saver.save(sess, "./dnn/my_model_final.ckpt")
```

    0 Batch accuracy: 0.94 Val accuracy: 0.911
    1 Batch accuracy: 1.0 Val accuracy: 0.9316
    2 Batch accuracy: 0.9 Val accuracy: 0.943
    3 Batch accuracy: 0.92 Val accuracy: 0.9484
    4 Batch accuracy: 0.92 Val accuracy: 0.9528
    5 Batch accuracy: 0.94 Val accuracy: 0.9556
    6 Batch accuracy: 0.98 Val accuracy: 0.957
    7 Batch accuracy: 0.94 Val accuracy: 0.961
    8 Batch accuracy: 0.98 Val accuracy: 0.9636
    9 Batch accuracy: 1.0 Val accuracy: 0.9662
    10 Batch accuracy: 0.98 Val accuracy: 0.966
    11 Batch accuracy: 0.98 Val accuracy: 0.9688
    12 Batch accuracy: 1.0 Val accuracy: 0.9702
    13 Batch accuracy: 0.96 Val accuracy: 0.9706
    14 Batch accuracy: 0.98 Val accuracy: 0.9724
    15 Batch accuracy: 0.98 Val accuracy: 0.972
    16 Batch accuracy: 1.0 Val accuracy: 0.9724
    17 Batch accuracy: 0.96 Val accuracy: 0.9752
    18 Batch accuracy: 1.0 Val accuracy: 0.9746
    19 Batch accuracy: 1.0 Val accuracy: 0.9736
    20 Batch accuracy: 0.98 Val accuracy: 0.9766
    21 Batch accuracy: 0.98 Val accuracy: 0.976
    22 Batch accuracy: 0.98 Val accuracy: 0.9778
    23 Batch accuracy: 0.98 Val accuracy: 0.976
    24 Batch accuracy: 1.0 Val accuracy: 0.976
    25 Batch accuracy: 1.0 Val accuracy: 0.9768
    26 Batch accuracy: 0.98 Val accuracy: 0.976
    27 Batch accuracy: 0.98 Val accuracy: 0.9774
    28 Batch accuracy: 1.0 Val accuracy: 0.9782
    29 Batch accuracy: 1.0 Val accuracy: 0.9772
    30 Batch accuracy: 0.98 Val accuracy: 0.9764
    31 Batch accuracy: 0.98 Val accuracy: 0.9778
    32 Batch accuracy: 0.98 Val accuracy: 0.9792
    33 Batch accuracy: 1.0 Val accuracy: 0.9776
    34 Batch accuracy: 0.98 Val accuracy: 0.9786
    35 Batch accuracy: 1.0 Val accuracy: 0.979
    36 Batch accuracy: 1.0 Val accuracy: 0.9782
    37 Batch accuracy: 1.0 Val accuracy: 0.9798
    38 Batch accuracy: 1.0 Val accuracy: 0.9792
    39 Batch accuracy: 1.0 Val accuracy: 0.9788


## 使用神经网络


```python
with tf.Session() as sess:
    saver.restore(sess, "./dnn/my_model_final.ckpt")
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

```

    INFO:tensorflow:Restoring parameters from ./dnn/my_model_final.ckpt



```python
print("Predicted classes:", y_pred)
print("Actual classes:", y_test[:20])
```

    Predicted classes: [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
    Actual classes: [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
****
[源文件](https://github.com/coldJune/machineLearning/blob/master/handson-ml/Introduction%20of%20Artificial%20Neural%20Networks.ipynb)
