---
title: pandas入门(二)
date: 2018-03-19 16:51:47
categories: true
copyright: true
tags:
    - pandas
    - 数据分析
description: 主要记录操作Series和DataFrame中的数据的基本手段。后面将更深入地挖掘pandas在数据分析和处理方面的功能
---
## 基本功能
### 重新索引

* reindex的(插值)method选项
|      参数       |        说明        |
|:---------------:|:------------------:|
|   fffill或pad   | 前向填充(或搬运)值 |
| bfill或backfill | 后向填充(或搬运)值 |

* reindex函数的参数
|    参数    |                                                       说明                                                       |
|:----------:|:----------------------------------------------------------------------------------------------------------------:|
|   index    | 用作索引的新序列。既可以是Index实例，也可以是其他序列型的Python数据结构。Index会被完全使用，就像没有任何复制一样 |
|   method   |                                                  插值(填充)方式                                                  |
| fill_value |                                 再重新索引的过程中，需要引入缺失值时使用的替代值                                 |
|   limit    |                                           前向或后向填充时的最大填充量                                           |
|   level    |                               在MultiIndex的指定级别上匹配简单索引，否则选取其子集                               |
|    copy    | 默认为True，无论如何都复制；如果为False，则新旧相等就不复制                                                                                                                 |
pandas对象的`reindex`方法用于创建一个适应新索引的新对象，`reindex`将会根据新索引进行重排。如果某个索引值当前不存在，就引入缺失值。`method`选项可以在重新索引时做一些插值处理：
```
In [86]: obj = Series([1,2,3,4],index=['a','b','c','d'])

In [87]: obj
Out[87]:
a    1
b    2
c    3
d    4
dtype: int64

In [88]: obj2 = obj.reindex(['q','w','e','r'])

In [89]: obj2
Out[89]:
q   NaN
w   NaN
e   NaN
r   NaN
dtype: float64

In [90]: obj2 = obj.reindex(['a','b','c','d','e'])

In [91]: obj2
Out[91]:
a    1.0
b    2.0
c    3.0
d    4.0
e    NaN
dtype: float64

In [94]: obj2 = obj.reindex(['a','b','c','d','e'],fill_value=0)

In [95]: obj2
Out[95]:
a    1
b    2
c    3
d    4
e    0
dtype: int64

In [98]: obj3 = obj.reindex(['a','b','e','f','c','d'],method='ffill')

In [99]: obj3
Out[99]:
a    1
b    2
e    4
f    4
c    3
d    4
dtype: int64
```

对于DataFrame,`reindex`可以修改(行)索引、列、或两个都修改。如果仅传入一个序列，则会重新索引行，使用`columns`关键字可以重新索引列,也可以同时对行和列进行重新索引，但插值只能按行应用(即轴0):
```
In [105]: frame = DataFrame(np.arange(9).reshape((3,3)),index=['a','b','c'],columns=['col1','col2','col3'])

In [106]: frame2 = frame.reindex(['a','b','c','d'])

In [107]: frame2
Out[107]:
   col1  col2  col3
a   0.0   1.0   2.0
b   3.0   4.0   5.0
c   6.0   7.0   8.0
d   NaN   NaN   NaN

In [108]: frame.reindex(columns=['col_a','col1','col2','col3'])
Out[108]:
   col_a  col1  col2  col3
a    NaN     0     1     2
b    NaN     3     4     5
c    NaN     6     7     8

In [109]: frame.reindex(index=['a','b','c','d'],method='ffill',columns=['col_a','col1','col2','col3'])
Out[109]:
   col_a  col1  col2  col3
a      2     0     1     2
b      5     3     4     5
c      8     6     7     8
d      8     6     7     8
```
利用ix的标签索引功能重新索引：
```
In [111]: frame.ix[['a','b','c','d'],['col_a','col1','col2','col3']]
Out[111]:
   col_a  col1  col2  col3
a    NaN   0.0   1.0   2.0
b    NaN   3.0   4.0   5.0
c    NaN   6.0   7.0   8.0
d    NaN   NaN   NaN   NaN
```
### 丢弃指定轴上的项
使用`drop`方法删除指定轴上的项，只需要传入一个索引数组或列表，对于DataFrame可以传入指定的轴(axis)来进行删除,返回的都是删除轴之后的新对象:
```
In [112]: obj = Series([1,2,3,4],index=['a','b','c','d'])

In [113]: obj.drop('a')
Out[113]:
b    2
c    3
d    4
dtype: int64

In [114]: obj.drop(['a','b'])
Out[114]:
c    3
d    4
dtype: int64

In [115]: frame = DataFrame(np.arange(9).reshape((3,3)),index=['a','b','c'],columns=['col1','col2','col3'])

In [116]: frame.drop(['a','b'])
Out[116]:
   col1  col2  col3
c     6     7     8

In [117]: frame.drop(['col1','col2'],axis=1)
Out[117]:
   col3
a     2
b     5
c     8
```

### 索引、选取和过滤
