---
title: pandas入门(二)
date: 2018-03-19 16:51:47
categories: 数据分析
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
|    copy    | 默认为True，无论如何都复制；如果为False，则新旧相等就不复制                                                                                                                |

pandas对象的`reindex`方法用于创建一个适应新索引的新对象，`reindex`将会根据新索引进行重排。如果某个索引值当前不存在，就引入缺失值。`method`选项可以在重新索引时做一些插值处理：
```Python
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

对于DataFrame,`reindex`可以修改(行)索引、列、或两个都修改。如果仅传入一个序列，则会重新索引行，使用`columns`关键字可以重新索引列,也可以同时对行和列进行重新索引，但插值只能按行应用:
```Python
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
利用`loc`的标签索引功能重新索引：
```Python
In [111]: frame.loc[['a','b','c','d'],['col_a','col1','col2','col3']]
Out[111]:
   col_a  col1  col2  col3
a    NaN   0.0   1.0   2.0
b    NaN   3.0   4.0   5.0
c    NaN   6.0   7.0   8.0
d    NaN   NaN   NaN   NaN
```
### 丢弃指定轴上的项
使用`drop`方法删除指定轴上的项，只需要传入一个索引数组或列表，对于DataFrame可以传入指定的轴(axis)来进行删除,返回的都是删除轴之后的新对象:
```Python
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
Series索引(obj[……])的工作方式类似于NumPy数组的索引，并且可以使用非整数；而利用切片运算其 **末端是包含的(封闭)**：
```Python
In [3]: obj = Series(np.arange(4), index=['a','b','c','d'])

In [4]: obj
Out[4]:
a    0
b    1
c    2
d    3
dtype: int64

In [5]: obj['a']
Out[5]: 0

In [6]: obj[2:4]
Out[6]:
c    2
d    3
dtype: int64

In [7]: obj['c':'d']
Out[7]:
c    2
d    3
dtype: int64

In [8]: obj[['a','d']]
Out[8]:
a    0
d    3
dtype: int64

In [9]: obj['b':'c']=5

In [10]: obj
Out[10]:
a    0
b    5
c    5
d    3
dtype: int64
```

对DataFrame进行索引是获取一个或多个列，可以通过切片或布尔型数组选取行，也可以使用布尔型DataFrame进行索引：
```Python
In [15]: data = DataFrame(np.arange(16).reshape(4,4),
    ...:                 index=['a','b','c','d'],
    ...:                 columns=['col1','col2','col3','col4'])
    ...:

In [16]: data
Out[16]:
   col1  col2  col3  col4
a     0     1     2     3
b     4     5     6     7
c     8     9    10    11
d    12    13    14    15

In [17]: data['col1']
Out[17]:
a     0
b     4
c     8
d    12
Name: col1, dtype: int64

In [18]: data[['col1','col4']]
Out[18]:
   col1  col4
a     0     3
b     4     7
c     8    11
d    12    15

In [19]: data[:2]
Out[19]:
   col1  col2  col3  col4
a     0     1     2     3
b     4     5     6     7

In [20]: data[data['col3']>5]
Out[20]:
   col1  col2  col3  col4
b     4     5     6     7
c     8     9    10    11
d    12    13    14    15

In [21]: data<5
Out[21]:
    col1   col2   col3   col4
a   True   True   True   True
b   True  False  False  False
c  False  False  False  False
d  False  False  False  False

In [22]: data[data<5] = -5

In [23]: data
Out[23]:
   col1  col2  col3  col4
a    -5    -5    -5    -5
b    -5     5     6     7
c     8     9    10    11
d    12    13    14    15
```
为了在DataFrame的行上进行标签索引，可以通过`loc`进行：
```Python
In [48]: data.loc['a',['col1','col2']]
Out[48]:
col1   -5
col2   -5
Name: a, dtype: int64

In [49]: data.loc[['a','d'],['col1','col3']]
Out[49]:
   col1  col3
a    -5    -5
d    12    14

In [50]: data.loc[data.col3>5,:'col3']
Out[50]:
   col1  col2  col3
b    -5     5     6
c     8     9    10
d    12    13    14
```

### 算术运算和数据对齐
pandas可以对不同索引的对象进行算数运算。在将对象相加时，如果存在不同的索引对，则结果的索引就是对该索引对的并集，自动的数据对齐操作在不重叠的索引处引入NA值，缺失值会在算术运算过程中传播:
```Python
In [55]: s1 = Series(np.arange(3),index=['a','b','c'])

In [56]: s2 = Series(np.arange(3,9),index=['a','b','c','d','e','f'])

In [57]: s1
Out[57]:
a    0
b    1
c    2
dtype: int64

In [58]: s2
Out[58]:
a    3
b    4
c    5
d    6
e    7
f    8
dtype: int64

In [59]: s1+s2
Out[59]:
a    3.0
b    5.0
c    7.0
d    NaN
e    NaN
f    NaN
dtype: float64
```
对于DataFrame，对齐操作会同时发生在行和列上，它们相加后会返回一个新的DataFrame，其索引和列为原来两个DataFrame的并集：
```Python
In [65]: df1 = DataFrame(np.arange(9).reshape(3,3),columns=list('abc'),
    ...:                 index=['row1','row2','row3'])
    ...:

In [66]: df2 = DataFrame(np.arange(16).reshape(4,4),columns=list('abcd'),
    ...:                 index=['row1','row2','row3','row4'])
    ...:

In [67]: df1
Out[67]:
      a  b  c
row1  0  1  2
row2  3  4  5
row3  6  7  8

In [68]: df2
Out[68]:
       a   b   c   d
row1   0   1   2   3
row2   4   5   6   7
row3   8   9  10  11
row4  12  13  14  15

In [69]: df1+df2
Out[69]:
         a     b     c   d
row1   0.0   2.0   4.0 NaN
row2   7.0   9.0  11.0 NaN
row3  14.0  16.0  18.0 NaN
row4   NaN   NaN   NaN NaN
```
#### 在算术方法中填充值
* 灵活的算术方法

| 方法 |        说明       |
| :--: | :---------------: |
| add  | 用于加法(+)的方法 |
| sub  | 用于减法(-)的方法 |
| div  | 用于除法(/)的方法 |
| mul  | 用于乘法(*)的方法                  |

对于不同索引的对象进行算术运算时，当一个对象中某个轴标签在另一个对象中找不到时填充一个特殊值,在对Series或DataFrame重新索引时也可以指定一个填充值：
```Python
In [76]: df2.add(df1,fill_value=0)
Out[76]:
         a     b     c     d
row1   0.0   2.0   4.0   3.0
row2   7.0   9.0  11.0   7.0
row3  14.0  16.0  18.0  11.0
row4  12.0  13.0  14.0  15.0

In [77]: df1.reindex(columns=df2.columns,fill_value=0)
Out[77]:
      a  b  c  d
row1  0  1  2  0
row2  3  4  5  0
row3  6  7  8  0
```

#### DataFrame和Series之间的运算
默认情况下DataFrame和Series之间的算术运算会将Series的索引匹配到DataFrame的列，然后沿着行一直向下广播；如果某个索引值在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引形成并集；如果希望匹配行且在列上广播则必须使用算术运算方法：
```Python
In [94]: s1 = df2.loc['row1']

In [95]: df2
Out[95]:
       a   b   c   d
row1   0   1   2   3
row2   4   5   6   7
row3   8   9  10  11
row4  12  13  14  15

In [96]: s1
Out[96]:
a    0
b    1
c    2
d    3
Name: row1, dtype: int64

In [97]: df2-s1
Out[97]:
       a   b   c   d
row1   0   0   0   0
row2   4   4   4   4
row3   8   8   8   8
row4  12  12  12  12

In [98]: s2 = Series(range(3),index=list('abf'))

In [99]: df2-s2
Out[99]:
         a     b   c   d   f
row1   0.0   0.0 NaN NaN NaN
row2   4.0   4.0 NaN NaN NaN
row3   8.0   8.0 NaN NaN NaN
row4  12.0  12.0 NaN NaN NaN

In [100]: s3  = df2['a']

Out[101]:
      a  b  c  d
row1  0  1  2  3
row2  0  1  2  3
row3  0  1  2  3
row4  0  1  2  3
```

### 函数应用和映射
NumPy的[ufuncs](http://coldjune.com/2018/03/17/numpy%E5%9F%BA%E7%A1%80-%E4%BA%8C/#%E9%80%9A%E7%94%A8%E5%87%BD%E6%95%B0)(元素级数组方法)也可用于操作pandas对象:
```Python
In [102]: frame = DataFrame(np.random.randn(4,3),columns=list('abc'),
     ...:                   index=['row1','row2','row3','row4'])
     ...:

In [103]: frame
Out[103]:
             a         b         c
row1  0.755289  0.886977 -0.984527
row2  0.460170 -0.514393  0.180462
row3  0.828386 -0.545317 -1.176786
row4  0.860822 -1.659938  0.952070

In [104]: np.abs(frame)
Out[104]:
             a         b         c
row1  0.755289  0.886977  0.984527
row2  0.460170  0.514393  0.180462
row3  0.828386  0.545317  1.176786
row4  0.860822  1.659938  0.952070

```

`apply`方法可以将函数应用到各列或行所形成的一维数组上，许多常见的数组统计功能都被实现成DataFrame方法(如sum和mean)，因此无需使用`apply`方法；除标量外，传递给`apply`的函数还可以返回多个值组成的Series；元素级的Python函数也是可以使用的，可以使用`applymap`得到frame中各个浮点值的格式化字符串:
```Python
In [112]: f = lambda x:x.max() -x.min()

In [113]: frame.apply(f)
Out[113]:
a    0.400653
b    2.546915
c    2.128856
dtype: float64

In [114]: def f(x):
     ...:     return Series([x.min(),x.max()],index=['min','max'])
     ...:

In [115]: frame.apply(f)
Out[115]:
            a         b         c
min  0.460170 -1.659938 -1.176786
max  0.860822  0.886977  0.952070

In [116]: format = lambda x: '%.2f' % x

In [117]: frame.applymap(format)
Out[117]:
         a      b      c
row1  0.76   0.89  -0.98
row2  0.46  -0.51   0.18
row3  0.83  -0.55  -1.18
row4  0.86  -1.66   0.95

In [118]: frame['a'].map(format)
Out[118]:
row1    0.76
row2    0.46
row3    0.83
row4    0.86
Name: a, dtype: object
```
### 排序和排名
#### 排序
使用`sort_index`方法对行或列索引进行排序(按字典顺序)，它将返回一个已排序的对象；对于DataFrame则可以根据任意一个轴上的索引进行排序；数据默认时按升序进行排序的，可以设置`ascending=False`来降序排序：
```Python
In [134]: obj = Series(range(4), index=list('dabc'))

In [135]: obj.sort_index()
Out[135]:
a    1
b    2
c    3
d    0
dtype: int64

In [136]: frame = DataFrame(np.arange(8).reshape((2,4)),index=['col2','col1'],
     ...:                    columns=list('badc'))
     ...:

In [137]: frame.sort_index()
Out[137]:
      b  a  d  c
col1  4  5  6  7
col2  0  1  2  3

In [138]: frame.sort_index(axis=1)
Out[138]:
      a  b  c  d
col2  1  0  3  2
col1  5  4  7  6

In [139]: frame.sort_index(axis=1, ascending=False)
Out[139]:
      d  c  b  a
col2  2  3  0  1
col1  6  7  4  5
```
`sort_values`方法用于按值进行排序，在排序时，任何的缺失值默认都会放到Series的末尾：
```Python
In [144]: obj.sort_values()
Out[144]:
4   -3.0
5    2.0
0    4.0
2    7.0
1    NaN
3    NaN
dtype: float64
```
在DataFrame中，可以将一个或多个列的名字传递给by选项来根据一个或多个列中的值进行排序，要根据多个列进行排序，可以传入名称的列表：
```Python
In [150]: frame  = DataFrame({'b':[2,5,0,1],'a':[0,1,0,1]})

In [151]: frame
Out[151]:
   a  b
0  0  2
1  1  5
2  0  0
3  1  1

In [152]: frame.sort_values(by='b')
Out[152]:
   a  b
2  0  0
3  1  1
0  0  2
1  1  5

In [153]: frame.sort_values(by=['a','b'])
Out[153]:
   a  b
2  0  0
0  0  2
3  1  1
1  1  5
```
#### 排名
排名会增设一个排名值(从1开始，一直到数组中有效的数据的数量)，它可以根据某种规则破坏平级关系；`rank`是通过“为各组分配一个平均排名”的方式破坏平级关系[^1]。
* 排名用于破坏平级关系的method的选项

|   method  |                   说明                   |
| :-------: | :--------------------------------------: |
| 'average' | 默认：在相等分组中，为各个值分配平均排名 |
|   'min'   |          使用整个分组的最小排名          |
|   'max'   |          使用整个分组的最大排名          |
|  'first'  | 按值在原始数据中的出现顺序分配排名                                         |

按降序进行排名使用`ascending=False`，其他的相似:
```Python
In [9]: obj = Series([7,6,7,5,4,4,3])

In [10]: obj.rank()
Out[10]:
0    6.5
1    5.0
2    6.5
3    4.0
4    2.5
5    2.5
6    1.0
dtype: float64

In [11]: obj.rank(method='min')
Out[11]:
0    6.0
1    5.0
2    6.0
3    4.0
4    2.0
5    2.0
6    1.0
dtype: float64

In [12]: obj.rank(method='max')
Out[12]:
0    7.0
1    5.0
2    7.0
3    4.0
4    3.0
5    3.0
6    1.0
dtype: float64

In [13]: obj.rank(method='first')
Out[13]:
0    6.0
1    5.0
2    7.0
3    4.0
4    2.0
5    3.0
6    1.0
dtype: float64

In [9]: obj = Series([7,6,7,5,4,4,3])

In [10]: obj.rank()
Out[10]:
0    6.5
1    5.0
2    6.5
3    4.0
4    2.5
5    2.5
6    1.0
dtype: float64

In [11]: obj.rank(method='min')
Out[11]:
0    6.0
1    5.0
2    6.0
3    4.0
4    2.0
5    2.0
6    1.0
dtype: float64

In [12]: obj.rank(method='max')
Out[12]:
0    7.0
1    5.0
2    7.0
3    4.0
4    3.0
5    3.0
6    1.0
dtype: float64

In [13]: obj.rank(method='first')
Out[13]:
0    6.0
1    5.0
2    7.0
3    4.0
4    2.0
5    3.0
6    1.0
dtype: float64
```

DataFrame可以在行或列上计算排名:
```Python
In [15]: frame = DataFrame({'b':[1,3,-1],'a':[2,-1,-2],'c':[1,2,3]})

In [16]: frame
Out[16]:
   a  b  c
0  2  1  1
1 -1  3  2
2 -2 -1  3

In [17]: frame.rank(axis=0)
Out[17]:
     a    b    c
0  3.0  2.0  1.0
1  2.0  3.0  2.0
2  1.0  1.0  3.0

In [18]: frame.rank(axis=1)
Out[18]:
     a    b    c
0  3.0  1.5  1.5
1  1.0  3.0  2.0
2  1.0  2.0  3.0

```

### 带有重复值的轴索引
带有重复索引值的Series和DataFrame可以使用`is_unique`属性确认它是否唯一；对于带有重复值的索引，如果某个值对应多个值，则会返回一个Series(或DataFrame)；而对应单个值则返回一个标量(Series)：
```Python
In [19]: obj = Series(range(5),index=list('abbvd'))

In [20]: obj
Out[20]:
a    0
b    1
b    2
v    3
d    4
dtype: int32

In [21]: obj.index.is_unique
Out[21]: False

In [22]: obj['a']
Out[22]: 0

In [23]: obj['b']
Out[23]:
b    1
b    2
dtype: int32

In [24]: df = DataFrame(np.random.randn(4,3),index=['a','a','b','c'])

In [26]: df
Out[26]:
          0         1         2
a  2.139973  0.102242  0.366141
a -0.999559  0.324575 -0.808672
b  1.121435  1.508694  1.151597
c  0.610592  1.623871 -1.331131

In [27]: df.loc['c']
Out[27]:
0    0.610592
1    1.623871
2   -1.331131
Name: c, dtype: float64

In [28]: df.loc['a']
Out[28]:
          0         1         2
a  2.139973  0.102242  0.366141
a -0.999559  0.324575 -0.808672
```
[^1]: 破坏平级关系是指在两个相同的数之间确认先后顺序。使用average表示如果在数组中7排在第五位和第六位，则其排名为5.5。min则为min(5,6)为5；max则为max(5,6)为7；first则表示在原数据中先出现排序靠前，紧邻的+1，依次递增。
