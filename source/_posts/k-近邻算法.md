---
title: k-近邻算法
date: 2018-05-15 10:03:05
categories: 机器学习
copyright: true
mathjax: true
tags:
    - 机器学习
    - 机器学习实战
    - 分类
    - 监督学习
description:
---

## K-近邻算法概述
> k-近邻算法采用测量不同特征值之间的距离方法进行分类

### k-近邻算法优缺点
* 优点：精度高、对异常值不敏感、无数据输入假定
* 缺点：计算复杂度高、空间复杂度高
* 使用数据范围：数值型和标称型

### 工作原理
> 存在一个样本数据集合且样本集中每个数据存在标签。输入没有标签的新数据后将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据(最近邻)的分类标签。只选取数据前k个最相似的数据。

### k-近邻算法的一般流程
1. 收集数据：使用任何方法
2. 准备数据：距离计算所需要的数值，最好是结构化的数据结构
3. 分析数据：可以使用任何方法
4. 训练算法：不适用与k-近邻算法
5. 测试算法：计算错误率
6. 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据所属类别，对计算出的分类执行后续处理

## 实现k-近邻算法
k-近邻算法是使用欧式距离公式计算两个向量点$xA$和$xB$之间的距离：
> $$d=\sqrt{(xA_0-xB_0)^2+(xA_1-xB_1)^2}$$

例如点$(0,0)$与$(1,2)$之间的距离为：
> $$\sqrt{(1-0)^2+(2-0)^2}$$

如果数据集存在4个特征值，则点$(1,2,3,4)$和$(2,3,4,5)$之间的距离计算为:
> $$\sqrt{(1-2)^2+(2-3)^2+(3-4)^2+(4-5)^2}$$

* k-近邻算法

```Python
def classify(in_x, data_set, labels, k):
    """k-近邻算法
    :param in_x: 分类的输入向量
    :param data_set: 输入的训练样本集
    :param labels: 标签向量，元素数目和矩阵dataSet的行数相同
    :param k: 用于选择最近邻居的数目
    :return:
    """
    data_set_size = data_set.shape[0]
    # 计算输入向量与样本的差值
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 计算欧式距离
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 排序
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        # 获取第i个元素的label
        vote_i_label = labels[sorted_dist_indicies[i]]
        # 计算该类别的数目
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 对类别按值进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]
```

## k-近邻算法实战
### 在约会网站是使用k-近邻算法
* 将文本记录转换到NumPy

```Python
def file2matrix(filename):
    """将文本记录转换为NumPy
    :param filename: 文件名
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as file:
        # 读取文本计算样本数量
        array_o_lines = file.readlines()
        number_of_lines = len(array_o_lines)
        # 生成样本举证
        return_mat = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_o_lines:
            # 处理每一个样本
            line = line.strip()
            list_from_line = line.split('\t')
            # 获取数据
            return_mat[index, :] = list_from_line[0:3]
            # 获取标签
            class_label_vector.append(int(list_from_line[-1]))
            index += 1
        return return_mat, class_label_vector
```
* 归一化数值

> 将任意取值范围的特征值转化为0~1区间内的值$new_{value} = \frac{old_{value}-min}{max-min}$

```Python
def auto_norm(data_set):
    """归一化特征值
    :param data_set: 数据集
    :return:
    """
    # 计算最小值和最大值及两者的差值
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    m = data_set.shape[0]
    # 归一化数据集
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals
```
* 测试

```Python
def dating_class_test():
    """分类器针对约会网站的测试代码
    :return:
    """
    ho_radtio = 0.10
    dating_data_mat, dating_labels = file2matrix('dataSet/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_radtio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                     dating_labels[num_test_vecs:m], 3)
        print('预测值为：%d,真实值为：%d' % (classifier_result, dating_labels[i]))
        if(classifier_result != dating_labels[i]):
            error_count += 1.0
    print("错误率为：%f" % (error_count/float(num_test_vecs)))

```

### 手写系统识别
* 将图像转换为测试向量
> 把一个32x32的二进制图像矩阵转换为1x1024的向量

```Python
def img2vector(filename):
    """将图像转换为测试向量
    将测试数据中32*32的二进制图像矩阵转换为1*1024的向量
    :param filename: 文件名
    :return:
    """
    return_vect = np.zeros((1, 1024))
    with open(filename, 'r', encoding='utf-8') as file:
        for i in range(32):
            line_str = file.readline()
            for j in range(32):
                return_vect[0, 32*i+j] = int(line_str[j])
    return return_vect
```

* 识别手写数字

```Python
def handwriting_class_test():
    """使用k-近邻算法识别手写数字
    :return:
    """
    hw_labels = []
    training_file_list = os.listdir('dataSet/digits/trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        # 加载数据集并添加标签
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('dataSet/digits/trainingDigits/%s' % file_name_str)

    test_file_list = os.listdir('dataSet/digits/testDigits')
    error_account = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        # 预测训练集
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        vector_under_test = img2vector('dataSet/digits/testDigits/%s' % file_name_str)
        classifier_result = classify(vector_under_test, training_mat, hw_labels, 3)
        print('预测值为：%d,真实值为：%d' % (classifier_result, hw_labels[i]))
        if(classifier_result != class_num_str):
            error_account += 1.0
    print("预测错误个数为:%d" % error_account)
    print("错误率为：%f" % (error_account/float(m_test)))
```
****
[示例代码](https://github.com/coldJune/machineLearning/blob/master/MachineLearningInAction/kNN/kNN.py)
