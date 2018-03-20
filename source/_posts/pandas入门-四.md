---
title: pandas入门(四)
date: 2018-03-20 15:43:20
categories: pandas
copyright: true
tags:
    - 数据分析
    - pandas
description: pandas入门主题的最后一点内容，关于层次化索引和整数索引及面板数据
---
## 层次化索引
层次化索引能在一个轴上拥有多个(两个以上)索引级别，能以低纬度形式处理高纬度数据。在创建Series时，可以使用一个由列表或数组组成的列表作为索引。对于一个层次化索引的对象，选取数据子集的操作同样很简单，有时可以在"内层"中进行选取：
```
In [206]: data = Series(np.random.randn(10),index=[list('aaabbbvvdd'),
     ...:            ['in1','in2','in3','in1','in2','in3','in1','in2','in2','in3']])
     ...:

In [207]: data
Out[207]:
a  in1    0.837994
   in2    0.360445
   in3   -0.657047
b  in1    0.017681
   in2   -0.577803
   in3    0.080992
v  in1   -0.158913
   in2   -0.011517
d  in2    0.632189
   in3   -1.181628
dtype: float64

In [208]: data['a']
Out[208]:
in1    0.837994
in2    0.360445
in3   -0.657047
dtype: float64

In [209]: data[:,'in1']
Out[209]:
a    0.837994
b    0.017681
v   -0.158913
dtype: float64
```
层次化索引在数据重塑和基于分组的操作中非常重要，使用`unstack`方法可以将Series多层索引安排到一个DataFrame中,`statck`是其逆运算:
```
In [210]: data.unstack()
Out[210]:
        in1       in2       in3
a  0.837994  0.360445 -0.657047
b  0.017681 -0.577803  0.080992
d       NaN  0.632189 -1.181628
v -0.158913 -0.011517       NaN

In [211]: data.unstack().stack()
Out[211]:
a  in1    0.837994
   in2    0.360445
   in3   -0.657047
b  in1    0.017681
   in2   -0.577803
   in3    0.080992
d  in2    0.632189
   in3   -1.181628
v  in1   -0.158913
   in2   -0.011517
dtype: float64
```
对于一个DataFrame，每条轴都可以有分层索引，各层都可以有名字；有了列索引后可以通过其选取列分组：
```
In [213]: df = DataFrame(np.arange(16).reshape(4,4),
     ...:                 index = [['row1','row1','row2','row2'],[1,2,1,2]],
     ...:                 columns=[['col1','col1','col2','col2'],['red','blue','red','blue']])
     ...:

In [214]: df
Out[214]:
       col1      col2
        red blue  red blue
row1 1    0    1    2    3
     2    4    5    6    7
row2 1    8    9   10   11
     2   12   13   14   15

In [215]: df.index.names=['rowname1','rowname2']

In [216]: df.columns.names=['colname1','colname2']

In [217]: df
Out[217]:
colname1          col1      col2
colname2           red blue  red blue
rowname1 rowname2
row1     1           0    1    2    3
         2           4    5    6    7
row2     1           8    9   10   11
         2          12   13   14   15

In [218]: df['col1']
Out[218]:
colname2           red  blue
rowname1 rowname2
row1     1           0     1
         2           4     5
row2     1           8     9
         2          12    13
```
### 重排分级顺序
* **swaplevel**
`swaplevel`接收两个级别编号或名称，并返回一个互换了级别的新对象：
```
In [219]: df
Out[219]:
colname1          col1      col2
colname2           red blue  red blue
rowname1 rowname2
row1     1           0    1    2    3
         2           4    5    6    7
row2     1           8    9   10   11
         2          12   13   14   15

In [220]: df.swaplevel('rowname1','rowname2')
Out[220]:
colname1          col1      col2
colname2           red blue  red blue
rowname2 rowname1
1        row1        0    1    2    3
2        row1        4    5    6    7
1        row2        8    9   10   11
2        row2       12   13   14   15
```

* **sort_index(level=)**
`sort_index(level=)`根据单个级别中的值对数据进行排序(稳定的):
```
In [225]: df.sort_index(level=1)
Out[225]:
colname1          col1      col2
colname2           red blue  red blue
rowname1 rowname2
row1     1           0    1    2    3
row2     1           8    9   10   11
row1     2           4    5    6    7
row2     2          12   13   14   15

In [226]: df.swaplevel(0,1).sort_index(level=0)
Out[226]:
colname1          col1      col2
colname2           red blue  red blue
rowname2 rowname1
1        row1        0    1    2    3
         row2        8    9   10   11
2        row1        4    5    6    7
         row2       12   13   14   15
```

### 根据级别汇总统计
许多对于DataFrame和Series的描述和汇总统计都有一个level选项，用于指定在某条轴上求和的级别：
```
In [225]: df.sort_index(level=1)
Out[225]:
colname1          col1      col2
colname2           red blue  red blue
rowname1 rowname2
row1     1           0    1    2    3
row2     1           8    9   10   11
row1     2           4    5    6    7
row2     2          12   13   14   15

In [226]: df.swaplevel(0,1).sort_index(level=0)
Out[226]:
colname1          col1      col2
colname2           red blue  red blue
rowname2 rowname1
1        row1        0    1    2    3
         row2        8    9   10   11
2        row1        4    5    6    7
         row2       12   13   14   15
```

### 使用DataFrame的列
* **set_index**
`set_index`函数将一个或多个列转换为行索引，并创建一个新的DataFrame，默认情况下用于创建索引的列会被移除，可以通过设置`drop=False`保留：
```
In [231]: frame = DataFrame({'a':range(7),'b':range(7,0,-1),
     ...:                 'c':['one','one','one','two','two','two','two'],
     ...:                 'd':[0,1,2,0,1,2,3]})
     ...:

In [232]: frame
Out[232]:
   a  b    c  d
0  0  7  one  0
1  1  6  one  1
2  2  5  one  2
3  3  4  two  0
4  4  3  two  1
5  5  2  two  2
6  6  1  two  3

In [233]: frame.set_index(['c','d'])
Out[233]:
       a  b
c   d
one 0  0  7
    1  1  6
    2  2  5
two 0  3  4
    1  4  3
    2  5  2
    3  6  1

In [234]: frame.set_index(['c','d'],drop=False)
Out[234]:
       a  b    c  d
c   d
one 0  0  7  one  0
    1  1  6  one  1
    2  2  5  one  2
two 0  3  4  two  0
    1  4  3  two  1
    2  5  2  two  2
    3  6  1  two  3
```

* **reset_index**
`reset_index`将层次化索引的级别转移到列里面，和`set_index`相反:
```
In [236]: frame2
Out[236]:
       a  b
c   d
one 0  0  7
    1  1  6
    2  2  5
two 0  3  4
    1  4  3
    2  5  2
    3  6  1

In [237]: frame2.reset_index()
Out[237]:
     c  d  a  b
0  one  0  0  7
1  one  1  1  6
2  one  2  2  5
3  two  0  3  4
4  two  1  4  3
5  two  2  5  2
6  two  3  6  1
```

## 整数索引
当一个pandas对象含有类似0、1、2的索引时，很难推断出需要的是基于标签或位置的索引，为了保证良好的一致性，如果轴索引含有索引器，那么根据整数进行数据选取的操作将总是面向标签的；如果需要可靠地、不考虑索引类型的、基于位置的索引，可以使用`loc`:
```
In [271]: obj = Series(np.arange(3))

In [272]: obj.loc[:1]
Out[272]:
0    0
1    1
dtype: int32

In [273]: frame = DataFrame(np.arange(9).reshape(3,3),index=[2,0,1])

In [274]: frame.loc[0,:]
Out[274]:
0    3
1    4
2    5
Name: 0, dtype: int32

In [275]: frame.loc[:,0]
Out[275]:
2    0
0    3
1    6
Name: 0, dtype: int32
```
