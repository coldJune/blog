---
title: 'House Prices: Advanced Regression Techniques'
date: 2019-05-25 08:57:50
categories: 机器学习
copyright: true
mathjax: true
tags: 
    - kaggle
description: 这是我第一个真正意义上完成的机器学习项目
---
# 前言
&nbsp;&nbsp;我真正接触Kaggle是在做《Hands-On Machine Learning with Scikit-Learn and TensorFlow》的一道练习题的时候，那道练习题使用的数据是Kaggle上一个分类数据集——[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)，当我登录这个页面的时候发现这是一个非常热门的项目，其参与团队(个人)已经达到了11223个，这对我这样一个初来乍到的人是一个不小的冲击，抱着决定在这个平台试一试的心态我开始寻找适合我的项目。
&nbsp;&nbsp;[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)是Kaggle上的一个知识性竞赛，作为一个回归问题，它提供了适量的数据以及特征供学习者使用；而作为机器学习的入门项目它帮助了很多人完成了从0到1的过程，现在上面有4746个团队(个人)提交了自己的预测结果。我作为一名学习者，也通过自己的努力在上面获得了自己的分数——0.12702，这是使用`KernelRidge`实现的模型进行预测的结果，这并不算一个很好的评分，大概排在1757名左右(前37%)，但对我来说确实一个很大的进步，这标示着从无到有的过程。
&nbsp;&nbsp;kaggle对于数据初学者来说确实是一个非常适合的平台，kaggler们都不吝啬自己的知识，发布着自己的kernel，表述自己的想法，借此帮助每一个需要帮助的社区成员。能完成这个项目对我来说意义非凡，在这里我特别感谢kaggle上的两位kaggler以及他们的对自己项目的无私奉献，他们分别是[@Pedro Marcelino](https://www.kaggle.com/pmarcelino)和他的kenel[Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python),他对数据的分析以及把控让现在的我难以望其项背，给了我非常大的启发；以及[@Serigne](https://www.kaggle.com/serigne)和他的kernel[Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard),他同样用了[@Pedro Marcelino](https://www.kaggle.com/pmarcelino)的数据分析方法，但是他在数据分析的基础上增加了模型的训练以及分析过程，帮助我学会把控自己的模型。再次对他们表示真挚的感谢。
&nbsp;&nbsp;虽然这个项目的准确程度还可以有很大的提升，但就我现在的能力而言我决定让它暂且休息一下，好回头看看，总结总结得失。

# 准备工作

## 准备数据
在准备工作阶段首先应该观察数据，并对数据做一定的处理
### 观察整个数据集


```python
# 导入相关数据包
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
```


```python
data = pd.read_csv('./house_price/train.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
data = data.set_index(['Id'])
```


```python
fig = plt.figure(figsize=(24, 36))
count = 1
for x in data[data.columns[data.dtypes != 'object']]:
    ax = fig.add_subplot(8,5, count)
    ax.scatter(y=data['SalePrice'], x=data[x])
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel('SalePrice', fontsize=13)
    ax.set_title(x)
    count += 1
plt.subplots_adjust(hspace=0.9, bottom=0.1, wspace=0.4)
```


![png](Predict%20House%20Prices_files/Predict%20House%20Prices_5_0.png)



```python
# data.drop(data[data['LotFrontage'] > 300].index, inplace=True)
# data.drop(data[data['LotArea'] > 100000].index, inplace=True)
# data.drop(data[data['MasVnrArea'] > 1500].index, inplace=True)
data.drop(data[(data['GrLivArea'] > 4000) & (data['SalePrice']<300000)].index, inplace=True)
# data.drop(data[data['BsmtFullBath'] == 3].index, inplace=True)
# data.drop(data[data['EnclosedPorch'] > 400].index, inplace=True)
# data.drop(data[data['PoolArea'] > 200].index, inplace=True)
# data.drop(data[data['MiscVal'] > 5000].index, inplace=True)
# fig = plt.figure(figsize=(24, 36))
# count = 1
# for x in data[data.columns[data.dtypes != 'object']]:
#     ax = fig.add_subplot(8,5, count)
#     ax.scatter(y=data['SalePrice'], x=data[x])
#     ax.set_xlabel(x, fontsize=13)
#     ax.set_ylabel('SalePrice', fontsize=13)
#     ax.set_title(x)
#     count += 1
# plt.subplots_adjust(hspace=0.9, bottom=0.1, wspace=0.4)
```


```python
#查看售价分布
sbn.distplot(data['SalePrice'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x114462860>




![png](Predict%20House%20Prices_files/Predict%20House%20Prices_7_1.png)



```python
sbn.distplot(np.log1p(data['SalePrice']))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119522048>




![png](House-Prices-Advanced-Regression-Techniques/Predict%20House%20Prices_8_1.png)



```python
x_train, y_train = data.loc[:,:'SaleCondition'], np.log1p(data['SalePrice']).get_values()
x_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1458 entries, 1 to 1460
    Data columns (total 79 columns):
    MSSubClass       1458 non-null int64
    MSZoning         1458 non-null object
    LotFrontage      1199 non-null float64
    LotArea          1458 non-null int64
    Street           1458 non-null object
    Alley            91 non-null object
    LotShape         1458 non-null object
    LandContour      1458 non-null object
    Utilities        1458 non-null object
    LotConfig        1458 non-null object
    LandSlope        1458 non-null object
    Neighborhood     1458 non-null object
    Condition1       1458 non-null object
    Condition2       1458 non-null object
    BldgType         1458 non-null object
    HouseStyle       1458 non-null object
    OverallQual      1458 non-null int64
    OverallCond      1458 non-null int64
    YearBuilt        1458 non-null int64
    YearRemodAdd     1458 non-null int64
    RoofStyle        1458 non-null object
    RoofMatl         1458 non-null object
    Exterior1st      1458 non-null object
    Exterior2nd      1458 non-null object
    MasVnrType       1450 non-null object
    MasVnrArea       1450 non-null float64
    ExterQual        1458 non-null object
    ExterCond        1458 non-null object
    Foundation       1458 non-null object
    BsmtQual         1421 non-null object
    BsmtCond         1421 non-null object
    BsmtExposure     1420 non-null object
    BsmtFinType1     1421 non-null object
    BsmtFinSF1       1458 non-null int64
    BsmtFinType2     1420 non-null object
    BsmtFinSF2       1458 non-null int64
    BsmtUnfSF        1458 non-null int64
    TotalBsmtSF      1458 non-null int64
    Heating          1458 non-null object
    HeatingQC        1458 non-null object
    CentralAir       1458 non-null object
    Electrical       1457 non-null object
    1stFlrSF         1458 non-null int64
    2ndFlrSF         1458 non-null int64
    LowQualFinSF     1458 non-null int64
    GrLivArea        1458 non-null int64
    BsmtFullBath     1458 non-null int64
    BsmtHalfBath     1458 non-null int64
    FullBath         1458 non-null int64
    HalfBath         1458 non-null int64
    BedroomAbvGr     1458 non-null int64
    KitchenAbvGr     1458 non-null int64
    KitchenQual      1458 non-null object
    TotRmsAbvGrd     1458 non-null int64
    Functional       1458 non-null object
    Fireplaces       1458 non-null int64
    FireplaceQu      768 non-null object
    GarageType       1377 non-null object
    GarageYrBlt      1377 non-null float64
    GarageFinish     1377 non-null object
    GarageCars       1458 non-null int64
    GarageArea       1458 non-null int64
    GarageQual       1377 non-null object
    GarageCond       1377 non-null object
    PavedDrive       1458 non-null object
    WoodDeckSF       1458 non-null int64
    OpenPorchSF      1458 non-null int64
    EnclosedPorch    1458 non-null int64
    3SsnPorch        1458 non-null int64
    ScreenPorch      1458 non-null int64
    PoolArea         1458 non-null int64
    PoolQC           6 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1458 non-null int64
    MoSold           1458 non-null int64
    YrSold           1458 non-null int64
    SaleType         1458 non-null object
    SaleCondition    1458 non-null object
    dtypes: float64(3), int64(33), object(43)
    memory usage: 911.2+ KB


### 数值型数据


```python
x_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1458.000000</td>
      <td>1199.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1450.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>...</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.893004</td>
      <td>69.797331</td>
      <td>10459.936900</td>
      <td>6.093964</td>
      <td>5.576132</td>
      <td>1971.218107</td>
      <td>1984.834019</td>
      <td>102.753793</td>
      <td>438.827160</td>
      <td>46.613169</td>
      <td>...</td>
      <td>472.050069</td>
      <td>94.084362</td>
      <td>46.245542</td>
      <td>21.984225</td>
      <td>3.414266</td>
      <td>15.081619</td>
      <td>2.433471</td>
      <td>43.548697</td>
      <td>6.323045</td>
      <td>2007.816187</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.329437</td>
      <td>23.203458</td>
      <td>9859.198156</td>
      <td>1.376369</td>
      <td>1.113359</td>
      <td>30.193754</td>
      <td>20.641760</td>
      <td>179.442156</td>
      <td>432.969094</td>
      <td>161.420729</td>
      <td>...</td>
      <td>212.239248</td>
      <td>125.350021</td>
      <td>65.312932</td>
      <td>61.155666</td>
      <td>29.337173</td>
      <td>55.792877</td>
      <td>38.209947</td>
      <td>496.460799</td>
      <td>2.700167</td>
      <td>1.328826</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7544.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>331.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9475.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1972.500000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>382.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>479.500000</td>
      <td>0.000000</td>
      <td>24.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11600.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>164.750000</td>
      <td>711.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>2188.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>1390.000000</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 36 columns</p>
</div>




```python
fig = plt.figure(figsize=(24, 24))
count = 1
for x in x_train[x_train.columns[x_train.dtypes != 'object']]:
    ax = fig.add_subplot(8,5, count)
    ax.boxplot(x_train[x])
    ax.set_title(x)
    count += 1
```

    /usr/local/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:1246: RuntimeWarning: invalid value encountered in less_equal
      wiskhi = np.compress(x <= hival, x)
    /usr/local/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:1253: RuntimeWarning: invalid value encountered in greater_equal
      wisklo = np.compress(x >= loval, x)
    /usr/local/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:1261: RuntimeWarning: invalid value encountered in less
      np.compress(x < stats['whislo'], x),
    /usr/local/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:1262: RuntimeWarning: invalid value encountered in greater
      np.compress(x > stats['whishi'], x)



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_12_1.png)



```python
x_train.hist(figsize=(24, 24))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x117c66748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11af6b470>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11af4e6d8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11aed6940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11aedb0b8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11aebc630>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x11ac11ba8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x118889860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x118889898>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11ace1c50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11ae2a208>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11afc0a20>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1188c9cf8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11b0b99e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11b095828>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11b061a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11ad6af98>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11ac68550>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x11b107ac8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fa1b080>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fa445f8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fa6bb70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fa9e128>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fac56a0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x11faefc18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fb1d1d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fb48748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fb6ecc0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fba0278>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fbc67f0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x11fbeed68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fc1e320>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fc48898>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fc6fe10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fca43c8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11fcc9940>]],
          dtype=object)




![png](Predict%20House%20Prices_files/Predict%20House%20Prices_13_1.png)


通过对数值型数据的各项指标的描述和直方图可以得出数据分布看出部分数据缺失，如`MSSubClass`、`LotFrontage`等，而大多数数据存在偏斜分布， 如`2ndFlrSF`、`3SsnPorch`等，对于缺失的数据，可以使用中位数进行填充，对于数据分布偏斜的问题，可以通过数据规范化进行调整


```python
plt.figure(figsize=(24, 24))
sbn.heatmap(x_train.corr(), linewidths=0.5, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b0dd898>




![png](Predict%20House%20Prices_files/Predict%20House%20Prices_15_1.png)


通过相关性矩阵可以看出`YearBuilt`和`GarageYrBlt`、`TotRmsAbvGrd`和`GrLiveArea`、`1stFlrSF`和`TotalBsmtSF`、`GarageCars`和`GarageArea`具有很高的相关性

### 非数值型数据
数值型数据无论是填充缺失值还是做规整化都是比较容易的，但非数值型数据的分析就稍显复杂了，首先是要确定非数值型数据的取值，然后是明晰每个取值的分布情况，即数量关系


```python
fig = plt.figure(figsize=(24, 48))
count = 1
for x in x_train.columns[x_train.dtypes == 'object']:
    ax = fig.add_subplot(15, 3, count)
    temp_feature = x_train[x].value_counts()
    feature_bar = ax.bar(range(temp_feature.shape[0]), temp_feature.values,  align='center')
    ax.set_xticks(np.arange(temp_feature.shape[0]))
    if temp_feature.shape[0] > 10:
        indexs = [index[-2:] for index in temp_feature.index]
        ax.set_xticklabels(indexs)
    else:
        ax.set_xticklabels(temp_feature.index)
    for bar in feature_bar:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2-0.1, 1.1*height, str(height))
#     ax.set_ylim(0, 1.2 * temp_feature.values[0])
    ax.set_title(x+'('+str(np.sum(temp_feature))+')')
    count+=1
    
plt.subplots_adjust(hspace=0.9, bottom=0.1)

    
```


![png](Predict%20House%20Prices_files/Predict%20House%20Prices_18_0.png)


通过柱状图可以观测出每个特征的具体数量以及特征中对应类别值的对应分布，从数据的描述中可以得到部分数据为`NA`的表示不具备相关特征，所以使用`None`填充，其它有缺失数据的特征可以使用当前特征中数量最多的类别进行填充；其中`Utilities`中共有1457条相同数据，1条相异数据，对模型并没有多少帮助，因此可以删去这个特征

## 处理数据
在这个阶段，结合以上对数据的观察与分析，着手对数据的预处理，包括填充空值，处理类别数据，筛选特征等


```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import learning_curve
class FeaturePreProcessing(BaseEstimator, TransformerMixin):
    """预处理所有特征"""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['MSSubClass'] = X['MSSubClass'].astype('str')
        X['YrSold'] = X['YrSold'].astype('str')
        X['MoSold'] = X['MoSold'].astype('str')
        X['OverallQual'] = X['OverallQual'].astype('str')
        X['OverallCond'] = X['OverallCond'].astype('str')
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        return X

class FeatureSelect(BaseEstimator, TransformerMixin):
    """特征选取"""
    def __init__(self, obj=True):
        self.obj = obj
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[X.columns[X.dtypes == 'object']] if self.obj else X[X.columns[X.dtypes != 'object']]
    
class NumericalImputer(BaseEstimator, TransformerMixin):
    """数值型特征填充空值"""
    def fit(self, X, y=None):
        self.attributes = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 
                           'Fireplaces', 
                           'MasVnrArea',
                           'GarageCars', 'GarageArea', 'GarageYrBlt']
        return self
    def transform(self, X, y=None):
        for attribute in self.attributes:
            X[attribute].fillna(0.0, inplace=True)
        # 添加总面积特征
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        return X

class StringImputer(BaseEstimator, TransformerMixin):
    """填充String类型的空值"""
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                        index=X.columns)
        return self
    
    def transform(self, X, y=None):
        """
        Alley: Type of alley access to property
           Grvl Gravel
           Pave Paved
           NA  No alley access
        """
        X['Alley'].fillna('None', inplace=True)
        """
        MasVnrType: Masonry veneer type
           BrkCmn   Brick Common
           BrkFace  Brick Face
           CBlock   Cinder Block
           None None
           Stone    Stone
        """
        X['MasVnrType'].fillna('None', inplace=True)
        """
        BsmtCond: Evaluates the general condition of the basement
           Ex   Excellent
           Gd   Good
           TA   Typical - slight dampness allowed
           Fa   Fair - dampness or some cracking or settling
           Po   Poor - Severe cracking, settling, or wetness
           NA   No Basement
        """
        X['BsmtCond'].fillna('None', inplace=True)
        """
        BsmtExposure: Refers to walkout or garden level walls
           Gd   Good Exposure
           Av   Average Exposure (split levels or foyers typically score average or above)  
           Mn   Mimimum Exposure
           No   No Exposure
           NA   No Basement
        """
        X['BsmtExposure'].fillna('None', inplace=True)
        """
        BsmtFinType1: Rating of basement finished area
           GLQ  Good Living Quarters
           ALQ  Average Living Quarters
           BLQ  Below Average Living Quarters   
           Rec  Average Rec Room
           LwQ  Low Quality
           Unf  Unfinshed
           NA   No Basement
        """
        X['BsmtFinType1'].fillna('None', inplace=True)
        """
        BsmtFinType2: Rating of basement finished area (if multiple types)
           GLQ  Good Living Quarters
           ALQ  Average Living Quarters
           BLQ  Below Average Living Quarters   
           Rec  Average Rec Room
           LwQ  Low Quality
           Unf  Unfinshed
           NA   No Basement
        """
        X['BsmtFinType2'].fillna('None', inplace=True)
        """
        FireplaceQu: Fireplace quality
           Ex   Excellent - Exceptional Masonry Fireplace
           Gd   Good - Masonry Fireplace in main level
           TA   Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
           Fa   Fair - Prefabricated Fireplace in basement
           Po   Poor - Ben Franklin Stove
           NA   No Fireplace
        """
        X['FireplaceQu'].fillna('None', inplace=True)
        """
        GarageType: Garage location
           2Types   More than one type of garage
           Attchd   Attached to home
           Basment  Basement Garage
           BuiltIn  Built-In (Garage part of house - typically has room above garage)
           CarPort  Car Port
           Detchd   Detached from home
           NA   No Garage
        """
        X['GarageType'].fillna('None', inplace=True)
        """
        GarageFinish: Interior finish of the garage
           Fin  Finished
           RFn  Rough Finished  
           Unf  Unfinished
           NA   No Garage
        """
        X['GarageFinish'].fillna('None', inplace=True)
        """
        GarageQual: Garage quality
           Ex   Excellent
           Gd   Good
           TA   Typical/Average
           Fa   Fair
           Po   Poor
           NA   No Garage
        """
        X['GarageQual'].fillna('None', inplace=True)
        """
        GarageCond: Garage condition
           Ex   Excellent
           Gd   Good
           TA   Typical/Average
           Fa   Fair
           Po   Poor
           NA   No Garage
        """
        X['GarageCond'].fillna('None', inplace=True)
#         X['GarageYrBlt'].fillna('None', inplace=True)
        """  
        PoolQC: Pool quality
           Ex Excellent
           Gd Good
           TA Average/Typical
           Fa Fair
           NA No Pool
        """
        X['PoolQC'].fillna('None', inplace=True)
        """
        Fence: Fence quality
           GdPrv  Good Privacy
           MnPrv  Minimum Privacy
           GdWo  Good Wood
           MnWw Minimum Wood/Wire
           NA No Fence
        """
        X['Fence'].fillna('None', inplace=True)
        """
        MiscFeature: Miscellaneous feature not covered in other categories
           Elev Elevator
           Gar2 2nd Garage (if not described in garage section)
           Othr Other
           Shed Shed (over 100 SF)
           TenC Tennis Court
           NA None
        """
        X['MiscFeature'].fillna('None', inplace=True)
        return X.fillna(self.most_frequent_)
    
class DropFeature(BaseEstimator, TransformerMixin):
    """删除部分特征"""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.features, axis=1)

class RemoveOutlier(BaseEstimator, TransformerMixin):
    """处理异常值"""
    def fit(self, X, y=None):
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.upper = q3 + 1.5 * iqr
        self.down = q1 - 1.5 * iqr
        self.median = X.median()
        return self
        
    def transform(self, X, y=None):
        X.where(X <= self.upper, self.upper, axis=1, inplace=True)
        X.where(X >= self.down, self.down, axis=1, inplace=True)
#         X['MiscVal'].where(X['MiscVal'] <= 5000, 5000, inplace=True)
#         X['LotFrontage'].where(X['LotFrontage'] <= 300, 300, inplace=True)
#         X['LotArea'].where(X['LotArea'] <= 100000, 100000, inplace=True )
#         X['MasVnrArea'].where(X['MasVnrArea'] <= 1500, 1500, inplace=True)
#         X['GrLivArea'].where(X['GrLivArea'] <= 4000, 4000, inplace=True)
#         X['EnclosedPorch'].where(X['EnclosedPorch'] <= 400, 400, inplace=True)
#         X['MiscVal'].where(X['MiscVal'] <= 5000, 5000, inplace=True)
        return X
    
def plot_learning_curve(model, X, y):
    train_size, train_scores, test_scores = learning_curve(model, X, y, 
                                                           n_jobs=-1, verbose=True, cv=10, random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color='r')
    plt.fill_between(train_size, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color='b')
    plt.plot(train_size, train_scores_mean, 'r-', label='train')
    plt.plot(train_size, test_scores_mean, 'b--',label='val')
    plt.ylim(0.5, 1.05)
    plt.yticks( np.linspace(0.5, 1, 11))
    plt.xlabel('Train Size', fontsize=14)
    plt.ylabel('acc', fontsize=14)
    plt.legend(loc='lower right')
```


```python
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

all_pipeline = Pipeline([
    ('featurePre', FeaturePreProcessing())
])
numeric_pipeline = Pipeline([
#         ('drop', DropFeature(['PoolQC','YearBuilt', 'TotRmsAbvGrd', '1stFlrSF'])),
        ('selector', FeatureSelect(False)),
        ('impute1', NumericalImputer()),
#         ('outlier', RemoveOutlier()),
        ('impute', SimpleImputer(strategy='median')),
        ('standard', StandardScaler())
])

cat_pipeline = Pipeline([
        ('drop', DropFeature([ 'Utilities'])),
        ('selector', FeatureSelect()),
        ('impute', StringImputer()),
        ('oneHot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

full_pipeline = Pipeline([
    ('all_pipeline', all_pipeline),
    ('featureunion',FeatureUnion([
        ('numeric_pipeline', numeric_pipeline),
        ('cat_pipeline', cat_pipeline)]))
])

x_train = full_pipeline.fit_transform(x_train)
```

# 模型选择
这里将尝试数个模型，比较它们的性能，来选择最优的模型

## 随机梯度下降


```python
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
# lr_pipline = Pipeline([
# #     ('poly', PolynomialFeatures(degree=2)),
#     ('lr', SGDRegressor(alpha=0.000, penalty='elasticnet'))
# ])
# lr_pipline.fit(x_train, y_train)
# lr_pred = cross_val_predict(lr_pipline, x_train, y_train, 
#                             verbose=True, n_jobs=-1, cv=3)
# lr_mse = mean_squared_error(lr_pred, y_train)
# np.sqrt(lr_mse)
```


```python
# plot_learning_curve(lr_pipline, x_train, y_train)
```

## 岭回归


```python
from sklearn.kernel_ridge import KernelRidge

ridge = KernelRidge(degree=2, alpha=0.05, kernel='polynomial')
ridge_pred = cross_val_predict(ridge, x_train, y_train, 
                               cv=3, verbose=True, n_jobs=-1)
ridge_mse = mean_squared_error(y_train, ridge_pred)
np.sqrt(ridge_mse)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:    1.5s finished





    0.11650866851364311




```python
plot_learning_curve(ridge, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    1.3s finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_29_3.png)


使用岭回归可以看到训练正确率有所下降，验证正确率有所上升，下面再试试`LASSO`回归

## LASSO回归


```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.0005)
lasso_pred = cross_val_predict(lasso, x_train, y_train, 
                               cv=3, verbose=True, n_jobs=-1)
lasso_mse = mean_squared_error(lasso_pred, y_train)
np.sqrt(lasso_mse)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:    0.2s finished





    0.1165640368257087




```python
plot_learning_curve(lasso, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    2.5s finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_33_3.png)


可以看出`Lasso`的效果并没有岭回归好，可能是因为`Lasso`使用`l1`范数稀疏掉了过多的特征导致其泛化能力的下降

## Elastic Net


```python
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV

# en = ElasticNet()
# en_param = {
#     'l1_ratio': np.linspace(0.3, 0.8),
#     'alpha': np.linspace(0, 0.01)
# }
# en_grid_cv = RandomizedSearchCV(en, param_distributions=en_param, verbose=True, 
#                                 cv=3, n_jobs=-1)
# en_grid_cv.fit(x_train, y_train)
en = ElasticNet(alpha=0.0012244897959183673, copy_X=True, fit_intercept=True,
      l1_ratio=0.3, max_iter=1000, normalize=False, positive=False,
      precompute=False, random_state=None, selection='cyclic', tol=0.0001,
      warm_start=False)
en.fit(x_train, y_train)
```




    ElasticNet(alpha=0.0012244897959183673, copy_X=True, fit_intercept=True,
          l1_ratio=0.3, max_iter=1000, normalize=False, positive=False,
          precompute=False, random_state=None, selection='cyclic', tol=0.0001,
          warm_start=False)




```python
# en_grid_cv.best_estimator_
```


```python
# en_grid_cv.best_score_
```


```python
en_pred = cross_val_predict(en, x_train, y_train, 
                            verbose=True, n_jobs=-1, cv=3)
mse = mean_squared_error(y_train, en_pred)
np.sqrt(mse)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:    0.3s finished





    0.11718105973133436




```python
plot_learning_curve(en, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    2.9s finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_40_3.png)


### KernelPCA


```python
# from sklearn.decomposition import KernelPCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# lr_pipeline = Pipeline([
# #     ('kpca', KernelPCA()),
#     ('lr', LinearRegression())
# ])

# lr_param = {
# #     'kpca__gamma': np.linspace(0.001, 0.03, 10),
# #     'kpca__kernel': ['rbf', 'linear', 'poly'],
#     'lr__normalize': [False, True]
# }

# lr_grid_cv = RandomizedSearchCV(lr_pipeline, param_distributions=lr_param, cv=3, 
#                           verbose=True, n_jobs=-1, iid=True)
# lr_grid_cv.fit(x_train, y_train)
```


```python
# lr_grid_cv.best_params_
```


```python
# lr_grid_cv.best_score_
```


```python
# lr = lr_grid_cv.best_estimator_
# lr_pred = cross_val_predict(lr, x_train, y_train, 
#                             cv=3, verbose=True, n_jobs=-1)
# mse = mean_squared_error(y_train, lr_pred)
# np.sqrt(mse)
```


```python
# plot_learning_curve(lr, x_train, y_train)
```

可以明显看到，使用PCA之后训练集的准确率为`1`，这是明显的过拟合现象

## 随机森林


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# rf_pipeline = Pipeline([
# #     ('kpca', KernelPCA()),
#     ('rf', RandomForestRegressor(n_estimators=100, oob_score=True))
# ])
# rf_param = {
# #     'kpca__gamma': np.linspace(0.01, 0.10, 10),
# #     'kpca__degree': np.arange(1, 5),
# #     'kpca__kernel': ['rbf', 'linear', 'poly'],
#     'rf__n_estimators': np.arange(3000, 9000, 100),
#     'rf__max_depth': np.arange(3, 10),
# #     'rf__min_samples_leaf': np.linspace(1e-3, 0.5),
# #     'rf__min_samples_split': np.linspace(1e-3, 1)
# }
# rf_grid_cv = RandomizedSearchCV(rf_pipeline, param_distributions=rf_param,
#                          cv=3, verbose=True, n_jobs=-1)
# rf_grid_cv.fit(x_train, y_train)
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=9,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=8300, n_jobs=None,
           oob_score=True, random_state=None, verbose=0, warm_start=False)
rf.fit(x_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=9,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=8300, n_jobs=None,
               oob_score=True, random_state=None, verbose=0, warm_start=False)




```python
# rf_grid_cv.best_estimator_
```


```python
# rf_grid_cv.best_score_
```


```python
rf_pred = cross_val_predict(rf, x_train, y_train, 
                                 verbose=True, n_jobs=-1, cv=3)
mse = mean_squared_error(y_train, rf_pred)
np.sqrt(mse)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:  2.3min finished





    0.1439767651809211




```python
plot_learning_curve(rf, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 29.5min finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_53_3.png)


## GBDT


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# gbdt_param = {
#     'n_estimators': np.arange(1000, 5000, 100),
#     'max_depth': np.arange(2, 10),
#     'subsample': np.linspace(0.1, 1, 20),
#     'max_features':['auto', 'sqrt', 'log2']
# }

# gbdt_grid_cv = RandomizedSearchCV(GradientBoostingRegressor(n_estimators=100), param_distributions=gbdt_param, n_jobs=-1,
#                                  verbose=True, random_state=42, cv=3)
# gbdt_grid_cv.fit(x_train, y_train)
gbdt = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=4,
             max_features='auto', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=3900,
             n_iter_no_change=None, presort='auto', random_state=None,
             subsample=0.5736842105263158, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
gbdt.fit(x_train, y_train)
```




    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=4,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=3900,
                 n_iter_no_change=None, presort='auto', random_state=None,
                 subsample=0.5736842105263158, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False)




```python
# gbdt_grid_cv.best_estimator_
```


```python
# gbdt_grid_cv.best_score_
```


```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
# gbdt = gbdt_grid_cv.best_estimator_
gbdt_pred = cross_val_predict(gbdt, x_train, y_train,
                             verbose=True, n_jobs=-1, cv=5)
mse = mean_squared_error(y_train, gbdt_pred)
np.sqrt(mse)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   58.5s finished





    0.12245307114524771




```python
plot_learning_curve(gbdt, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  4.5min finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_59_3.png)


## XGBoost


```python
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
# xg = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
# xg_pipeline = Pipeline([
# #     ('kpca',KernelPCA()),
#     ('xg', xg)
# ])
# xg_param = {
# #     'kpca__gamma': np.linspace(0.01, 0.10, 10),
# #     'kpca__kernel': ['rbf', 'linear', 'poly'],
#     'xg__n_estimators': np.arange(3000, 5000, 100),
#     'xg__max_depth': np.arange(2, 10) ,
#     'xg__gamma': np.linspace(0, 0.1),
#     'xg__min_child_weight': np.linspace(1, 5),
#     'xg__reg_lambda': np.linspace(0, 1),
#     'xg__reg_alpha': np.linspace(0, 1),
#     'xg__colsample_bytree': np.linspace(0, 1),
#     'xg__subsample': np.linspace(0, 1),
#     'xg__learning_rate': np.linspace(0, 0.1)
# }
# xg_grid_cv = RandomizedSearchCV(xg_pipeline, param_distributions=xg_param, cv=3,
#                                n_jobs=-1, verbose=True)
# xg_grid_cv.fit(x_train, y_train)
xg = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
         colsample_bynode=1, colsample_bytree=0.673469387755102,
         gamma=0.030612244897959186, importance_type='gain',
         learning_rate=0.030612244897959186, max_delta_step=0, max_depth=8,
         min_child_weight=1.2448979591836735, missing=None,
         n_estimators=3600, n_jobs=-1, nthread=None, objective='reg:linear',
         random_state=42, reg_alpha=0.5714285714285714,
         reg_lambda=0.12244897959183673, scale_pos_weight=1, seed=None,
         silent=None, subsample=0.6938775510204082, verbosity=1)
xg.fit(x_train, y_train)
```

    [19:46:24] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=0.673469387755102,
           gamma=0.030612244897959186, importance_type='gain',
           learning_rate=0.030612244897959186, max_delta_step=0, max_depth=8,
           min_child_weight=1.2448979591836735, missing=None,
           n_estimators=3600, n_jobs=-1, nthread=None, objective='reg:linear',
           random_state=42, reg_alpha=0.5714285714285714,
           reg_lambda=0.12244897959183673, scale_pos_weight=1, seed=None,
           silent=None, subsample=0.6938775510204082, verbosity=1)




```python
#  xg_grid_cv.best_estimator_
```


```python
# xg_grid_cv.best_score_
```


```python
xg_pred = cross_val_predict(xg, x_train, y_train, 
                            cv=3, verbose=True, n_jobs=-1)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:  1.5min finished



```python
mse = mean_squared_error(y_train, xg_pred)
np.sqrt(mse)
```




    0.12419601284479917




```python
plot_learning_curve(xg, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 21.1min finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_66_3.png)


## stack 


```python
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold

class StackModel(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, base_models, final_model, n_folds=5):
        self.base_models = base_models
        self.final_model = final_model
        self.n_folds = n_folds
        
    def fit(self, X, y):
        self.base_models_ = [list() for i in self.base_models]
        self.final_model_ = clone(self.final_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        out_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_predictions[holdout_index, i] = y_pred
        
        self.final_model_.fit(out_predictions, y)
        return self
                
    def predict(self, X):
        final_feature = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.final_model_.predict(final_feature)
        
```


```python
from sklearn.linear_model import  Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

ridge = KernelRidge(degree=2, alpha=0.05, kernel='polynomial')
lasso = Lasso(alpha=0.0005)
en = ElasticNet(alpha=0.0012244897959183673, copy_X=True, fit_intercept=True,
      l1_ratio=0.3, max_iter=1000, normalize=False, positive=False,
      precompute=False, random_state=None, selection='cyclic', tol=0.0001,
      warm_start=False)
GBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=4,
             max_features='auto', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=3900,
             n_iter_no_change=None, presort='auto', random_state=None,
             subsample=0.5736842105263158, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)

stack_models = StackModel(base_models=(ridge, GBR, en), final_model=lasso)
stack_models.fit(x_train, y_train)
```




    StackModel(base_models=(KernelRidge(alpha=0.05, coef0=1, degree=2, gamma=None, kernel='polynomial',
          kernel_params=None), GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=4,
                 max_features='auto', max_leaf_nodes=None,...
          precompute=False, random_state=None, selection='cyclic', tol=0.0001,
          warm_start=False)),
          final_model=Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False),
          n_folds=5)




```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
stack_pred = stack_models.predict(x_train)
mse = mean_squared_error(stack_pred, y_train)
np.sqrt(mse)
```




    0.06843459950134186




```python
plot_learning_curve(stack_models, x_train, y_train)
```

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 16.8min finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_71_2.png)


## LightGBM


```python
from  lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
lgb_params = {
    'n_estimators': np.arange(500, 3000, 100),
    'num_leaves': np.arange(3, 100),
    'learning_rate': np.linspace(1e-3, 0.1),
    'max_bin': np.arange(10, 100),
    'bagging_fraction': np.linspace(1e-3, 1),
    'bagging_freq': np.arange(1, 10),
    'feature_fraction': np.linspace(1e-3, 1),
    'min_data_in_leaf': np.arange(1, 20),
    'min_sum_hessian_in_leaf': np.linspace(1e-3, 20),
    'max_depth': np.arange(10, 20)
}
lgb_grid_cv = RandomizedSearchCV(LGBMRegressor(objective='regression', feature_fraction_seed=42, bagging_seed=42), 
                                param_distributions=lgb_params, cv=5,
                                n_jobs=-1, verbose=True)
lgb_grid_cv.fit(x_train, y_train)
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   17.1s
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   18.4s finished





    RandomizedSearchCV(cv=5, error_score='raise-deprecating',
              estimator=LGBMRegressor(bagging_seed=42, boosting_type='gbdt', class_weight=None,
           colsample_bytree=1.0, feature_fraction_seed=42,
           importance_type='split', learning_rate=0.1, max_depth=-1,
           min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
           n_estimators=100, n_jobs=-1, num_leaves=31, objective='regression',
           random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
           subsample=1.0, subsample_for_bin=200000, subsample_freq=0),
              fit_params=None, iid='warn', n_iter=10, n_jobs=-1,
              param_distributions={'n_estimators': array([ 500,  600,  700,  800,  900, 1000, 1100, 1200, 1300, 1400, 1500,
           1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
           2700, 2800, 2900]), 'num_leaves': array([ 3,  4, ..., 98, 99]), 'learning_rate': array([0.001  , 0.00302, 0.00...91837e+01, 1.95919e+01, 2.00000e+01]), 'max_depth': array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring=None, verbose=True)




```python
lgb_grid_cv.best_score_
```




    0.9038911706779178




```python
lgb_grid_cv.best_params_
```




    {'num_leaves': 3,
     'n_estimators': 2300,
     'min_sum_hessian_in_leaf': 2.858,
     'min_data_in_leaf': 14,
     'max_depth': 13,
     'max_bin': 27,
     'learning_rate': 0.011102040816326531,
     'feature_fraction': 1.0,
     'bagging_freq': 5,
     'bagging_fraction': 0.775734693877551}




```python
lgb_pred = cross_val_predict(lgb_grid_cv.best_estimator_, x_train, y_train, 
                  cv=3, n_jobs=-1)
mse = mean_squared_error(y_train, lgb_pred)
np.sqrt(mse)
```




    0.12889382465830496




```python
plot_learning_curve(lgb_grid_cv.best_estimator_, x_train, y_train)
```

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   11.1s finished



![png](Predict%20House%20Prices_files/Predict%20House%20Prices_77_2.png)


## 结合不同预测


```python
ensemble_pred = stack_pred*0.8 + xg_pred*0.2
mse = mean_squared_error(ensemble_pred, y_train)
np.sqrt(mse)
```




    0.07707245327525736



# 预测


```python
test = pd.read_csv('house_price/test.csv')
index = np.array(test[['Id']])[:,0]
test = test.set_index(['Id'])
x_test = full_pipeline.transform(test)
```


```python
# ridge.fit(x_train, y_train)
pre =  lgb_grid_cv.best_estimator_.predict(x_test)
# xg_pre = xg.predict(x_test)
# ensemble_pred = stack_pre*0.8 + xg_pre*0.2
pred_df = pd.DataFrame({'Id':index,
                       'SalePrice':np.expm1(pre)})
pred_df.to_csv('./house_price/prediction.csv', index='')
```


```python

```


```python

```

