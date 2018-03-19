---
title: pandas入门
date: 2018-03-19 10:48:58
categories: true
copyright: true
tags:
    - 数据分析
    - pandas
description: pandas含有使数据分析工作变得更快更简单的高级数据结构和操作工具。
---
## Series
**Series**[^1] 是一种类似于一维数组的对象，它由一组数据(各种NumPy数据类型)以及一组与之相关的数据标签(即索引)组成。Series的字符串表现形式为：索引在左边，值在右边。如果没有为数据指定索引，会自动创建一个0到n-1的整数型索引。可以通过`index`参数指定索引来代替自动生成的索引:
```
In [4]: ser1 = Series([1,2,2,3])

In [5]: ser1
Out[5]:
0    1
1    2
2    2
3    3
dtype: int64

In [6]: ser2 = Series([1,2,2,3],index=['a','b','c','d'])

In [7]: ser2
Out[7]:
a    1
b    2
c    2
d    3
dtype: int64
```
可以通过索引的方式选取Series中的单个或一组值；数组运算(布尔型数组进行过滤，标量乘法，应用数学函数)都会保留索引和值之间的连接；Series可以看成是一个定长的有序字典，可以用在原本需要字典参数的函数中:
```
In [8]: ser2['a']
Out[8]: 1

In [9]: ser2[['a','b']]
Out[9]:
a    1
b    2
dtype: int64

In [10]: ser2*2
Out[10]:
a    2
b    4
c    4
d    6
dtype: int64

In [11]: ser2[ser2>=2]
Out[11]:
b    2
c    2
d    3
dtype: int64

In [12]: 'a' in ser2
Out[12]: True

In [13]: 'g' in ser2
Out[13]: False
```
可以直接通过字典来创建Series，则Series中的索引就是原字典的键(有序列表)，如果键对应的值找不到，将会是使用`NA`表示缺失数据,pandas的`isnull`和`notnull`函数可用于检测缺失数据：
```
In [14]: dic = {'a':1,'b':2,'c':3}

In [15]: dics = Series(dic)

In [16]: dics
Out[16]:
a    1
b    2
c    3
dtype: int64

In [17]: states = ['a','b','c','d']

In [18]: dicstates = Series(dic,index=states)

In [19]: dicstates
Out[19]:
a    1.0
b    2.0
c    3.0
d    NaN
```
**Series在算数运算中会自动对齐不同索引的数据**：
```
In [20]: dics
Out[20]:
a    1
b    2
c    3
dtype: int64

In [21]: dicstates
Out[21]:
a    1.0
b    2.0
c    3.0
d    NaN
dtype: float64

In [22]: dics+dicstates
Out[22]:
a    2.0
b    4.0
c    6.0
d    NaN
dtype: float64
```
Series本身及其索引有一个name属性，同时Series的索引可以通过赋值的方式就地修改:
```
In [23]: dics.name='dics'

In [24]: dics.index.name='letter'

In [25]: dics
Out[25]:
letter
a    1
b    2
c    3
Name: dics, dtype: int64

In [26]: dics.index=['z','x','y']

In [27]: dics
Out[27]:
z    1
x    2
y    3
Name: dics, dtype: int64
```
## DataFrame

[^1]:使用 from pandas import Series, DataFrame和 import pandas as pd引入相关的包
