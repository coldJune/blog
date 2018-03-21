---
title: pandas入门(三)
date: 2018-03-20 09:14:48
categories: 数据分析
copyright: true
tags:
    - 数据分析
    - pandas
description: 更高级的数学及数据处理方法
---
## 汇总和计算描述统计
pandas对象拥有一组常用的数学和统计方法。她们大部分属于约简和汇总统计，用于从Series中提取单个值(如sum或mean)或从DataFrame的行或列中提取一个Series，他们都是基于没有缺失数据的假设构建的。
* 约简方法的选项

|  选项  |                          说明                           |
|:------:|:-------------------------------------------------------:|
|  axis  |            约简的轴。DataFrame的行用0，列用1            |
| skipna |                排除缺失值，默认值为True                 |
| level  | 如果轴是层次化索引的(即MultiIndex)，则根据level分组约简 |

* 描述和汇总统计

|      方法      |                     说明                     |
|:--------------:|:--------------------------------------------:|
|     count      |                 非NA值的数量                 |
|    describe    |    针对Series或各DataFrame列计算汇总统计     |
|    min、max    |              计算最小值和最大值              |
| argmin、argmax | 计算能够获取到最小值和最大值的索引位置(整数) |
| idxmin、idmax  |     计算能够获取到最小值和最大值的索引值     |
|    quantile    |            计算样本的分位数(0到1)            |
|      sum       |                   值的总和                   |
|      mean      |                  值的平均值                  |
|     median     |                指的算术中位数                |
|      mad       |          根据平均值计算平均绝对离差          |
|      var       |                 样本值的方差                 |
|      std       |                样本值的标准差                |
|      skew      |             样本值的偏度(三阶矩)             |
|      kurt      |             样本值的峰度(四阶矩)             |
|     cumsum     |                样本值的累计和                |
| cummin、cummax |        样本值的累计最小值和累计最大值        |
|    cumprod     |                样本值的累计积                |
|      diff      |        计算一阶差分(对时间序列有用)_         |
|   pct_change   | 计算百分数变化                                             |
```
In [33]: df = DataFrame([[1,np.nan],[2,3],[np.nan,np.nan],[4,5]],
    ...:                 index=list('abcd'),
    ...:                 columns=['one','two'])
    ...:

In [34]: df
Out[34]:
   one  two
a  1.0  NaN
b  2.0  3.0
c  NaN  NaN
d  4.0  5.0

In [35]: df.sum()
Out[35]:
one    7.0
two    8.0
dtype: float64

In [36]: df.sum(axis=1)
Out[36]:
a    1.0
b    5.0
c    NaN
d    9.0
dtype: float64

In [37]: df.mean(axis=1,skipna=False)
Out[37]:
a    NaN
b    2.5
c    NaN
d    4.5
dtype: float64

In [38]: df.idxmax()
Out[38]:
one    d
two    d
dtype: object

In [39]: df.cumsum()
Out[39]:
   one  two
a  1.0  NaN
b  3.0  3.0
c  NaN  NaN
d  7.0  8.0
```
`describe`用于一次性产生多个汇总统计，对于非数值类型会产生另外一种汇总统计：
```
In [40]: df.describe()
Out[40]:
            one       two
count  3.000000  2.000000
mean   2.333333  4.000000
std    1.527525  1.414214
min    1.000000  3.000000
25%    1.500000  3.500000
50%    2.000000  4.000000
75%    3.000000  4.500000
max    4.000000  5.000000

In [41]: obj =Series(list('aabc')*4)

In [42]: obj.describe()
Out[42]:
count     16
unique     3
top        a
freq       8
dtype: object
```

### 相关系数和协方差
Series的`corr`方法用于计算两个Series中重叠的、非NAN的、按索引对齐的相关系数；使用`cov`计算协方差：
```
In [46]: obj = Series([1,2,3,4],index=list('abcd'))

In [47]: obj2 = Series([1,np.nan,5,6,7],index=list('acdse'))

In [48]: obj.corr(obj2)
Out[48]: 1.0

In [49]: obj.cov(obj2)
Out[49]: 6.0
```
DataFrame的`corr`和`cov`方法将以DataFrame的形式返回完整的相关系数或协方差矩阵：
```
In [60]: df = DataFrame(np.arange(16).reshape(4,4),
    ...:                 index=list('abcd'),
    ...:                 columns=['col1','col2','col3','col4'])
    ...:

In [61]: df2 = DataFrame(np.arange(25).reshape(5,5),
    ...:                 index=list('abcde'),
    ...:                 columns=['col1','col2','col3','col4','col5'])
    ...:
    ...:

In [62]: df
Out[62]:
   col1  col2  col3  col4
a     0     1     2     3
b     4     5     6     7
c     8     9    10    11
d    12    13    14    15

In [63]: df2
Out[63]:
   col1  col2  col3  col4  col5
a     0     1     2     3     4
b     5     6     7     8     9
c    10    11    12    13    14
d    15    16    17    18    19
e    20    21    22    23    24

In [64]: df.corr()
Out[64]:
      col1  col2  col3  col4
col1   1.0   1.0   1.0   1.0
col2   1.0   1.0   1.0   1.0
col3   1.0   1.0   1.0   1.0
col4   1.0   1.0   1.0   1.0

In [65]: df.cov()
Out[65]:
           col1       col2       col3       col4
col1  26.666667  26.666667  26.666667  26.666667
col2  26.666667  26.666667  26.666667  26.666667
col3  26.666667  26.666667  26.666667  26.666667
col4  26.666667  26.666667  26.666667  26.666667
```
利用DataFrame的`corrwith`方法可以计算其列或行跟另一个Series或DataFrame之间的相关系数；传入一个Series将会返回一个相关系数值Series，传入一个DataFrame则会计算按列名配对的相关系数(传入axis=1按行计算)：
```
In [66]: df.corrwith(df2)
Out[66]:
col1    1.0
col2    1.0
col3    1.0
col4    1.0
col5    NaN
dtype: float64

In [69]: df.corrwith(df2.col1)
Out[69]:
col1    1.0
col2    1.0
col3    1.0
col4    1.0
dtype: float64
```

###唯一值、值计数以及成员资格

* 唯一值、值计数、成员资格方法

|     方法     |                             说明                             |
|:------------:|:------------------------------------------------------------:|
|     isin     | 计算一个表示“Series各值是否包含于传入的值序列中”的布尔型数组 |
|    unique    |           计算Series中的唯一值数组，按发现顺序返回           |
| value_counts | 返回一个Series，其索引为唯一值，其值为频率，按计数值降序排列 |

`unique`可以从Series中获取唯一值数组，返回的唯一值是未排序的，可以对结果进行排序(`unique().sort()`)。`value_counts`用于计算一个Series中各值出现的频率，结果Series是按值频率降序排列的。`value_counts`是一个顶级pandas方法，可以用于任何数组或序列；`isin`用于判断矢量化集合的成员资格，可用于选取Series中或DataFrame列中数据的子集：
```
In [78]: obj = Series(list('abbddc'))

In [79]: sor  = obj.unique()

In [80]: sor
Out[80]: array(['a', 'b', 'd', 'c'], dtype=object)

In [81]: sor.sort()

In [82]: sor
Out[82]: array(['a', 'b', 'c', 'd'], dtype=object)

In [83]: obj.value_counts()
Out[83]:
b    2
d    2
c    1
a    1
dtype: int64

In [84]: pd.value_counts(obj.values, sort=False)
Out[84]:
d    2
a    1
b    2
c    1
dtype: int64

In [85]: mask = obj.isin(['a','c'])

In [86]: mask
Out[86]:
0     True
1    False
2    False
3    False
4    False
5     True
dtype: bool

In [87]: obj[mask]
Out[87]:
0    a
5    c
dtype: object
```
可以将`pandas.value_counts`传递给DataFrame的`aplly`函数得到DataFrame中多个相关列的柱状图：
```
In [89]: data = DataFrame({'Q1':[1,3,4,4,5],
    ...:                    'Q2':[2,3,4,2,1],
    ...:                     'Q3':[4,1,4,5,6]})
    ...:

In [90]: data
Out[90]:
   Q1  Q2  Q3
0   1   2   4
1   3   3   1
2   4   4   4
3   4   2   5
4   5   1   6

In [91]: result = data.apply(pd.value_counts).fillna(0)

In [92]: result
Out[92]:
    Q1   Q2   Q3
1  1.0  1.0  1.0
2  0.0  2.0  0.0
3  1.0  1.0  0.0
4  2.0  1.0  2.0
5  1.0  0.0  1.0
6  0.0  0.0  1.0
```

## 处理缺失数据
缺失数据在大部分数据分析应用中都很常见。pandas使用浮点值NaN(Not a Number)表示浮点和非浮点数组中的缺失数据，它只是一个便于检测的标记。Python内置的None值也会被当做NA处理

* NA处理方法

|  方法   |                                        说明                                         |
|:-------:|:-----------------------------------------------------------------------------------:|
| dropna  |    根据各标签中是否存在缺失数据对轴标签进行过滤，可通过阈值调节对缺失值的容忍度     |
| fillna  |                   用指定值或插值方法(如ffill或bfill)填充缺失数据                    |
| isnull  | 返回一个含有布尔值的对象，这些布尔值表示哪些值是缺失值/NA，该对象的类型与源类型一样 |
| notnull |                                   isnull的否定式                                    |

```
In [99]: obj = Series([1,np.nan,2,np.nan,4])

In [100]: obj.isnull()
Out[100]:
0    False
1     True
2    False
3     True
4    False
dtype: bool

In [101]: obj[0]=None

In [102]: obj.isnull()
Out[102]:
0     True
1     True
2    False
3     True
4    False
dtype: bool
```

### 滤除缺失数据
对于Series，`dropna`返回一个仅含有非空数据和索引值的Series(通过布尔型索引达到一样的效果)：
```
In [104]: obj
Out[104]:
0    NaN
1    NaN
2    2.0
3    NaN
4    4.0
dtype: float64

In [105]: obj.dropna()
Out[105]:
2    2.0
4    4.0
dtype: float64

In [106]: obj[obj.notnull()]
Out[106]:
2    2.0
4    4.0
dtype: float64
```
对于DataFrame对象，`dropna`默认丢弃任何含有缺失值的行，传入`how='all'`将只丢弃全为NA的那些行，要丢弃列需要传入`axis=1`
```
In [108]: data = DataFrame([[1,4,5],[1,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,2,3]])

In [109]: data
Out[109]:
     0    1    2
0  1.0  4.0  5.0
1  1.0  NaN  NaN
2  NaN  NaN  NaN
3  NaN  2.0  3.0

In [110]: data.dropna()
Out[110]:
     0    1    2
0  1.0  4.0  5.0

In [111]: data.dropna(how='all')
Out[111]:
     0    1    2
0  1.0  4.0  5.0
1  1.0  NaN  NaN
3  NaN  2.0  3.0

In [112]: data[3]=np.nan

In [113]: data
Out[113]:
     0    1    2   3
0  1.0  4.0  5.0 NaN
1  1.0  NaN  NaN NaN
2  NaN  NaN  NaN NaN
3  NaN  2.0  3.0 NaN

In [114]: data.dropna(axis=1,how='all')
Out[114]:
     0    1    2
0  1.0  4.0  5.0
1  1.0  NaN  NaN
2  NaN  NaN  NaN
3  NaN  2.0  3.0
```
`thresh`参数移除非NA个数小于设定值的行：
```
In [123]: df = DataFrame(np.random.randn(7,3))

In [124]: df.loc[:3,1] = np.nan

In [125]: df.loc[:2,2] = np.nan

In [126]: df.dropna(thresh=2)
Out[126]:
          0         1         2
3  0.620445       NaN -0.379638
4 -0.642811  0.033634  0.700009
5  0.510774  1.458027  1.247687
6  0.614596 -1.986715 -0.378179
```

### 填充缺失数据
`fillna`方法是填充缺失数据的主要函数。通过一个常数调用`fillna`将会将缺失值替换为那个常数值；通过字典调用`fillna`可以实现对不同的列填充不同的值；`fillna`默认会返回新对象，通过设置`inplace=True`可以对现有对象进行就地修改，对`reindex`有效的插值方法也可用于`fillna`:
* fillna函数的参数

|  参数   |                           说明                            |
|:-------:|:---------------------------------------------------------:|
|  value  |             用于填充缺失值的标量值或字典对象              |
| method  | 插值方式。如果函数调用时未指定其他参数的话，默认为“ffill” |
|  axis   |                  待填充的轴，默认axis=0                   |
| inplace |                修改调用者对象而不产生副本                 |
|  limit  |        (对于前向和后向填充)可以连续填充的最大数量         |

```
In [127]: df
Out[127]:
          0         1         2
0 -0.293799       NaN       NaN
1  0.728953       NaN       NaN
2  0.573023       NaN       NaN
3  0.620445       NaN -0.379638
4 -0.642811  0.033634  0.700009
5  0.510774  1.458027  1.247687
6  0.614596 -1.986715 -0.378179

In [128]: df.fillna(0)
Out[128]:
          0         1         2
0 -0.293799  0.000000  0.000000
1  0.728953  0.000000  0.000000
2  0.573023  0.000000  0.000000
3  0.620445  0.000000 -0.379638
4 -0.642811  0.033634  0.700009
5  0.510774  1.458027  1.247687
6  0.614596 -1.986715 -0.378179

In [129]: df.fillna({1:0.5, 3:-1})
Out[129]:
          0         1         2
0 -0.293799  0.500000       NaN
1  0.728953  0.500000       NaN
2  0.573023  0.500000       NaN
3  0.620445  0.500000 -0.379638
4 -0.642811  0.033634  0.700009
5  0.510774  1.458027  1.247687
6  0.614596 -1.986715 -0.378179

In [130]: _  = df.fillna(0,inplace=True)

In [131]: df
Out[131]:
          0         1         2
0 -0.293799  0.000000  0.000000
1  0.728953  0.000000  0.000000
2  0.573023  0.000000  0.000000
3  0.620445  0.000000 -0.379638
4 -0.642811  0.033634  0.700009
5  0.510774  1.458027  1.247687
6  0.614596 -1.986715 -0.378179

In [138]: df = DataFrame(np.random.randn(7,3))

In [139]: df.loc[3:,1] = np.nan

In [140]: df.loc[2:,2] = np.nan

In [141]: df
Out[141]:
          0         1         2
0 -1.741073 -0.993316 -1.030055
1  0.139948 -1.446029  0.797856
2 -0.373251  0.505183       NaN
3  1.179879       NaN       NaN
4  0.764752       NaN       NaN
5  1.405856       NaN       NaN
6 -1.053222       NaN       NaN

In [142]: df.fillna(method='ffill')
Out[142]:
          0         1         2
0 -1.741073 -0.993316 -1.030055
1  0.139948 -1.446029  0.797856
2 -0.373251  0.505183  0.797856
3  1.179879  0.505183  0.797856
4  0.764752  0.505183  0.797856
5  1.405856  0.505183  0.797856
6 -1.053222  0.505183  0.797856
```
