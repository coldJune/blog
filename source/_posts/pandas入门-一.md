---
title: pandas入门(一)
date: 2018-03-19 10:48:58
categories: pandas
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

### 构造DataFrame
* 可以输入给DataFrame构造器的数据

|             类型             |                                        说明                                        |
|:----------------------------:|:----------------------------------------------------------------------------------:|
|         二维ndarray          |                           数据矩阵，还可以传入行标和列标                           |
| 由数组、列标或元组组成的字典 |               每个序列会变成DataFrame的一列，所有序列的长度必须相同                |
|    NumPy的结构化/记录数组    |                              类似于“由数组组成的字典”                              |
|      由Series组成的字典      | 每个Series会成为一列。如果没有显示指定索引，则个Series的索引会被合并成结果的行索引 |
|       由字典组成的字典       |   各内层字典会成为一列。键会被合并成结果的行索引，跟“由Series组成的字典”情况一样   |
|      字典或Series的列表      |    各项将会成为DataFrame的一行。字典键或Series索引的并集将会成为DataFrame的列标    |
|       另一个DataFrame        |                该DataFrame的索引将会被沿用，除非显式指定了其他索引                 |
|      NumPy的MaskedArray      |        类似于“二维ndarray”的情况，只是掩码值在结果DataFrame会编程NA/缺失值         |

**DataFrame** 是一个表格型的数据结构。它含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔值等)。DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典(共同用一个索引)，DataFrame面向行和面向列的操作基本上是平衡的。
构建DataFrame可以通过直接传入一个由等长列表或NumPy数组组成的字典，和Series一样DataFrame也会自动加上索引且全部列会被有序排列，如果指定了列索引，则DataFrame的列会按照指定顺序进行排列。如果传入的列在数据中找不到，会产生NA值：
```
In [30]: data ={'state':['a','b','c','d'],
    ...: 'year':[2000,2001,2002,2003],
    ...: 'pop':[1,2,3,4]}

In [31]: frame = DataFrame(data)

In [32]: frame
Out[32]:
   pop state  year
0    1     a  2000
1    2     b  2001
2    3     c  2002
3    4     d  2003

In [34]: DataFrame(data,columns=['year','pop','state','debt'],index=['i1','i2','i3','i4'])
Out[34]:
    year  pop state debt
i1  2000    1     a  NaN
i2  2001    2     b  NaN
i3  2002    3     c  NaN
i4  2003    4     d  NaN

In [35]: frame.columns
Out[35]: Index(['pop', 'state', 'year'], dtype='object')
```
可以通过字典标记的方式或属性的方式将DataFrame的列获取为一个Series，返回的Series拥有原DataFrame相同的索引，且其`name`属性已经被相应地设置好了。行也可以通过位置或名称的方式进行获取，比如用索引字段ix:
```
In [40]: frame.state
Out[40]:
0    a
1    b
2    c
3    d
Name: state, dtype: object

In [41]: frame['year']
Out[41]:
0    2000
1    2001
2    2002
3    2003
Name: year, dtype: int64

In [42]: frame.ix[1]
Out[42]:
pop         2
state       b
year     2001
Name: 1, dtype: object
```
列可以通过赋值的方式进行修改，将列表或数组给某个列时，其长度必须跟DataFrame的长度相匹配。如果赋值的事一个Series就会精确匹配DataFrame的索引，所有的空位都将被填上缺失值，为不存在的列赋值会创建出一个新列，关键字`del`可以删除列:
```
In [49]: frame2=DataFrame(data,columns=['year','pop','state','debt'],index=['i1','i2','i3','i4'])

In [50]: frame2
Out[50]:
    year  pop state debt
i1  2000    1     a  NaN
i2  2001    2     b  NaN
i3  2002    3     c  NaN
i4  2003    4     d  NaN

In [51]: frame2['debt']=np.arange(4.)

In [52]: frame2
Out[52]:
    year  pop state  debt
i1  2000    1     a   0.0
i2  2001    2     b   1.0
i3  2002    3     c   2.0
i4  2003    4     d   3.0

In [53]: frame2=DataFrame(data,columns=['year','pop','state','debt'],index=['i1','i2','i3','i4'])

In [54]: frame2
Out[54]:
    year  pop state debt
i1  2000    1     a  NaN
i2  2001    2     b  NaN
i3  2002    3     c  NaN
i4  2003    4     d  NaN

In [55]: val = Series([-1,-2,-3],index=['i1','i3','i4'])

In [56]: frame2['debt']=val

In [57]: frame2
Out[57]:
    year  pop state  debt
i1  2000    1     a  -1.0
i2  2001    2     b   NaN
i3  2002    3     c  -2.0
i4  2003    4     d  -3.0

In [58]: frame2['big']= frame2['pop']>=3

In [59]: frame2
Out[59]:
    year  pop state  debt    big
i1  2000    1     a  -1.0  False
i2  2001    2     b   NaN  False
i3  2002    3     c  -2.0   True
i4  2003    4     d  -3.0   True

In [60]: del frame2['big']

In [61]: frame2
Out[61]:
    year  pop state  debt
i1  2000    1     a  -1.0
i2  2001    2     b   NaN
i3  2002    3     c  -2.0
i4  2003    4     d  -3.0
```
嵌套字典被传给DataFrame后会被解释为：外层字典的键作为列，内层字典键作为行索引，可以通过`T`进行转置。内层字典的键会被合并，排序以形成最终的索引。如果现实指定了索引，就不会如此。同理，Series组成的字典也是一样的用法:
```
In [63]: pop = {'out1':{2002:1.1,2001:1.2},
    ...: 'out2':{2001:1.3,2004:1.4}}

In [64]: frame3 = DataFrame(pop)

In [65]: frame3
Out[65]:
      out1  out2
2001   1.2   1.3
2002   1.1   NaN
2004   NaN   1.4

In [66]: frame3.T
Out[66]:
      2001  2002  2004
out1   1.2   1.1   NaN
out2   1.3   NaN   1.4

In [67]: DataFrame(pop,index=[2002,2001,2004])
Out[67]:
      out1  out2
2002   1.1   NaN
2001   1.2   1.3
2004   NaN   1.4

In [68]: sData = {'out1':frame3['out1'][:-1],
    ...: 'out2':frame3['out2'][:-1]}

In [69]: DataFrame(sData)
Out[69]:
      out1  out2
2001   1.2   1.3
2002   1.1   NaN
```
设置了DataFrame的`index`和`columns`的`name`属性，这些信息将会被显示出来，`values`属性会以二维ndarray的形式返回DataFrame中的数据，如果DataFrame各列的数据类型不同，则值数组的数据类型就会选用能兼容所有列的数据类型：
```
In [70]: frame3.index.name='year'

In [71]: frame3.columns.name='state'

In [72]: frame3
Out[72]:
state  out1  out2
year
2001    1.2   1.3
2002    1.1   NaN
2004    NaN   1.4

In [73]: frame3.values
Out[73]:
array([[ 1.2,  1.3],
       [ 1.1,  nan],
       [ nan,  1.4]])

In [74]: frame2.values
Out[74]:
array([[2000, 1, 'a', -1.0],
       [2001, 2, 'b', nan],
       [2002, 3, 'c', -2.0],
       [2003, 4, 'd', -3.0]], dtype=object)
```
### 索引对象
pandas的索引对象负责管理轴标签和其他元数据。
* pandas中主要的Index对象

|      类       |                                说明                                |
|:-------------:|:------------------------------------------------------------------:|
|     Index     |  最泛化的Index对象，将轴标签表示为一个由Python对象组成的NumPy数组  |
|  Int64Index   |                        针对整数的特殊Index                         |
|  MultiIndex   | "层次化"索引对象，表示单个轴上的多层索引。可以看做由元组组成的数组 |
| DatetimeIndex |           存储纳秒级时间戳(用NumPy的datetime64类型表示)            |
|  PeriodIndex  |                针对Period数据(时间间隔)的特殊Index                 |

* Index的方法和属性

|     方法     |                        说明                        |
|:------------:|:--------------------------------------------------:|
|    append    |       连接另一个Index对象，产生一个新的Index       |
|     diff     |             计算差集，并得到一个Index              |
| intersection |                      计算交集                      |
|    union     |                      计算并集                      |
|     isin     | 计算一个指示各值是否都包含在参数集合中的布尔型数组 |
|    delete    |         删除索引i处的元素，并得到新的Index         |
|     drop     |           删除传入的值，并得到新的Index            |
|    insert    |        将元素插入到索引i处，并得到新的Index        |
| is_monotonic |      当各元素均大于等于前一个元素时，返回True      |
|  is_unique   |           当Index没有重复值时，返回True            |
|    unique    |              计算Index中唯一值的数组               |

构建Series或DataFrame时，所得到的任何数组或其他序列的标签都会被转换成一个Index，Index对象是 **不可修改的**，这使得Index对象在多个数据结构之间安全共享。除了长得像数组，Index的功能也类似与一个固定大小的集合，每个索引都有一些方法和属性，它们用于设置逻辑并回答有关索引所包含数据的常见问题:
```
In [76]: obj = Series(range(3),index=['a','b','c'])

In [77]: index =obj.index

In [78]: index
Out[78]: Index(['a', 'b', 'c'], dtype='object')

In [79]: index[:-1]
Out[79]: Index(['a', 'b'], dtype='object')

In [80]: inde=pd.Index(np.arange(3))

In [81]: obj2=Series(['a','b','c'],index=inde)

In [82]: obj2.index is inde
Out[82]: True

In [83]: frame3
Out[83]:
state  out1  out2
year
2001    1.2   1.3
2002    1.1   NaN
2004    NaN   1.4

In [84]: 'out1' in frame3.columns
Out[84]: True

In [85]: 2005 in frame3.index
Out[85]: False
```

[^1]:使用 from pandas import Series, DataFrame和 import pandas as pd引入相关的包
