---
title: 'House Prices: Advanced Regression Techniques(2)'
date: 2019-06-27 19:30:00
categories: 机器学习
copyright: true
mathjax: true
tags:
    - kaggle
description: 接上一篇，主要记录集成模型的训练过程
---

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;书接上文，在前文中我们已经基本看到了机器学习需要经历的基本过程，其中占据大部分篇幅的是数据分析和处理的部分，模型的训练反而占比不大。这其中的原因除了因为特征工程在整个机器学习中应有如此大的比重之外，还因为之前训练的模型都是一些简单模型，并不涉及到大量参数的调试。而现在，我们使用的集成学习，将会涉及到不少的参数需要调节。
# 集成学习概述
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;既然都单独将这部分内容拿出来描述了，那么在正式进入调参之前，我们先简要看看集成学习是个什么东西吧。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**集成学习**通过构建并结合多个学习器来完成学习任务，其先产生一组“个体学习器”，再用某种策略将它们结合起来。集成的方式又分为**同质集成**和**异质集成**。**同质集成**只包含相同类型的个体学习器，其个体学习器也称为“基学习器”；**异质集成**中的个体学习器是不同类型的，其被称为“组件学习器”。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;根据个体学习器的生成方式可以分为两大类，一种是串行生成的序列化方法，其个体学习器之间存在强依赖关系，代表为*Boosting*；另一种是同时生成的并行化方法，其个体学习器之间不存在强依赖关系，代表有*Bagging*和*Random Forest*。*Boosting*是一族可将弱学习器提升为强学习器的算法，它先从初始训练集训练出一个基学习器，再根据其表现调整训练样本的分布，即重新分配每个样本的权重，使表现不好的样本在下次训练时得到更多的关注，直至达到预定的训练次数，最后将所有的基学习器进行加权结合；*Boosting*每一次都是使用的全量数据，而*Bagging*却并不是，它采用有放回的采样的方式来生成训练集，每个基学习器使用不同的训练集来进行训练，有放回的采样使得同一个数据集能够被多次使用训练出不同的模型，最后可以通过投票(分类)和平均(回归)来结合各个基学习器的结果；*Random Forest*是在*Bagging*的基础上进一步在决策树的训练过程中引入随机属性选择,传统的决策树选择划分属性时是在当前节点的属性集合中选择一个最优属性，而在*RF*中是先从该结点的属性集合中随机选择一个包含$k$个属性的子集，再从子集中选择最优属性。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;集成学习还有很多细节上的东西，包括**Boosting**和**Bagging**的训练过程，最后结果的结合方式等等，在这里就不再进行一一陈述了。下面让我们进入主题——集成模型的训练吧。

# 模型选择
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在进入正式的调参之前，我先创建了一个公用方法，其目的是为了画出网格搜索($GridSearch$)过程中的平均准确率和准确率的变异系数[^1]。这两个分数是比较简单的用于衡量一个模型好坏的指标，这里将测试和训练进行了对比展示，从而对模型的泛化能力进行评估，对参数选择做出决断。
```
def plot_acc_4_grid(grid_cv, param):
        fig = plt.figure(figsize=(10, 10))
        mean_acc = fig.add_subplot(2,1,1)
        std_acc = fig.add_subplot(2,1,2)
        # 训练参数个数
        params_num = len(grid_cv.cv_results_['params'])
        x_ticks = np.arange(params_num)
        # 把每一次的参数作为横坐标label
        score_label = [list(grid_cv.cv_results_['params'][i].values())[0] for i in range(params_num)]
        # 平均精确度
        mean_train_score = grid_cv.cv_results_['mean_train_score']
        mean_test_score =  grid_cv.cv_results_['mean_test_score']
        # 方差
        std_train_score = grid_cv.cv_results_['std_train_score']
        std_test_score = grid_cv.cv_results_['std_test_score']

        mean_acc.plot(mean_train_score, 'r-o', label='mean_train_score')
        mean_acc.plot(mean_test_score , 'b-o', label='mean_test_score')
        mean_acc.set_title('mean_acc@'+param, fontsize=18)
        mean_acc.set_xticks(x_ticks)
        mean_acc.set_xticklabels(score_label)
        mean_acc.set_xlabel(param, fontsize=18)
        mean_acc.set_ylabel('mean_acc', fontsize=18)
        mean_acc.legend(loc='best', fontsize=18)
        mean_acc.grid()

        std_acc.plot(std_train_score,'r-*', label='std_train_score')
        std_acc.plot(std_test_score, 'b-*', label='std_test_score')
        std_acc.set_title('std_acc@'+param, fontsize=18)
        std_acc.set_xticks(x_ticks)
        std_acc.set_xticklabels(score_label)
        std_acc.set_xlabel(param, fontsize=18)
        std_acc.set_ylabel('std_acc', fontsize=18)
        std_acc.legend(loc='best', fontsize=18)
        std_acc.grid()

        plt.subplots_adjust(hspace=0.5)
```

## 随机森林
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们先来尝试一下上面提到的随机森林，这里主要关注的是影响性能的几个参数，罗列如下：
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
        text-align: left;
        word-wrap:break-word;
    	word-break:break-all;
    	white-space:normal;
    	max-width:650px;
        font-family:SimSun;
    }

    .dataframe thead th {
        text-align: middle;
    }
</style>
<div style="text-align: center;">
    <table class='dataframe' style='margin: auto; width:100%'>
        <thead>
            <tr>
                <th>参数</th>
                <th>详情</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th>n_estimators</th>
                <th>子模型的数量<br>&nbsp;&nbsp;&nbsp;&nbsp;• integer(n_estimators≥1)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为10
                </th>
            </tr>
            <tr>
                <th>max_depth</th>
                <th>树的最大深度<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• integer(max_depth≥1)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• None(树会生长到所有叶子节点都分到一个类或者某节点所代表的样本数据已小于min_samples_split)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为None
                </th>
            </tr>
            <tr>
                <th>max_features</th>
                <th>在寻找最佳划分时考虑的最大特征数<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• integer(n_features≥max_features≥1)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• float(占所有特征的百分比)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• "auto"(n_features，即所有特征)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• "sqrt"(max_features=sqrt(n_features)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• "log2"(max_features=log2(n_features))<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• None(n_features，即所有特征)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为"auto"
                </th>
            </tr>
            <tr>
                <th>min_samples_split</th>
                <th>内部节点分裂所需的最小样本数<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• integer(min_samples_split≥2)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• float(ceil(min_samples_split * n_samples)，即占所有样本的百分比向下取整)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为2
                </th>
            </tr>
            <tr>
                <th>max_leaf_nodes</th>
                <th>最大叶节点数<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• integer(max_leaf_nodes≥1)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• None(不限制叶节点个数)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为None
                </th>
            </tr>
            <tr>
                <th>min_weight_fraction_leaf</th>
                <th>叶节点最小样本权重总值<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• float(权重总值)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为0
                </th>
            </tr>
            <tr>
                <th>min_samples_leaf</th>
                <th>叶节点最小样本数<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• integer(min_samples_leaf≥1)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• float(ceil(min_samples_leaf * n_samples)，即占所有样本的百分比向下取整)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为1
                </th>
            </tr>
            <tr>
                <th>bootstrap</th>
                <th>是否使用bootstrap对样本进行采样<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• False(所有子模型的样本一致，子模型强相关)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• True(每个子模型的样本从总样本中有放回采样)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为True
                </th>
            </tr>
            <tr>
                <th>criterion</th>
                <th>判断节点是否分裂的使用的计算方法<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• "mse"(均方误差)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;• "mae"(平均绝对误差)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;* 默认值为"mse"
                </th>
            </tr>
        </tbody>
    </table>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中的`n_estimators`的值一般来说是越大性能越好，泛化能力越强，是单调递增的，但随着子模型的数量增加，训练算法所消耗的资源和时间将会急剧增加。其它数值型参数对性能的影响都呈现出有增有减的，而枚举型的例如`criterion`则需要视情况而定了。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们已经对需要调节的参数有了一个直观的认识，知道每个参数代表的含义。但是我们怎么来调节他们呢？在一开始的时候，我选择了一种非常笨重的方式，我通过直接将所有参数塞进`GridSearchCV`，而这导致训练需要花费大量的时间，不如设想有*3*个参数需要调节，每个参数取*10*个待定值，最后需要尝试的组合高达**1000**个之多，而这里的参数有`9`个之多，如果是更复杂的神经网络，那基本上就是望山跑死马的事了。我后知后觉得意识到网格查找的局限性，又想起书里提到的随机方法`RandomizedSearchCV`，我马上进行了尝试，并一度以为这样就能完美解决问题，但显然我是过于乐观了——最后训练出的模型基本上都是过拟合的，而且参数可控性极低。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这里最后尝试的方法是一种基于贪心策略的坐标下降法，即每一次只调节一个参数，然后选择最优的参数固定下来继续训练下一个参数，这可以大大减少训练所需的资源和时间——将上面的**1000**减少到**30**，只要能保证每个参数对性能的提升都是单调递增的，就能取得不错的效果。下面让我们来一探究竟吧。

* n_estimators

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先进行调节的是`n_estimators`这个参数，正如前文所述，这是一个使性能单调递增的参数，我们首先在粗粒度对它训练，观察训练的整体趋势。
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rf_param = {
    'n_estimators': np.arange(1, 1000, 100),
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=50), param_grid=rf_param,
                         cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 10 candidates, totalling 30 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:  1.8min finished
    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([  1, 101, 201, 301, 401, 501, 601, 701, 801, 901])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;下面是这*10*个模型得分的最终情况，我们可以训练得分和验证得分有比较大的差距，但是两者的提升都是同步的，不曾出现此消彼长的情况；在子模型数为*1~101*时两个得分的提高最为明显，之后增加子模型数量对分数并未有任何贡献，虽然如此，但是通过观察变异系数发现验证集的得分的离散程度还有一定幅度的减小，这意味子模型数量的增加虽然无异于提高分数，但是其使模型更加稳定，得分更加一致。
```
plot_acc_4_grid(rf_grid_cv, 'n_estimators')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_60_0.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面进行了粗粒度的训练，但是这样大刀阔斧地调参会忽视掉许多细节，比如是否两个取值点刚好错过了可能存在之间的波峰或者波谷，结果是否存在波动等。基于此我们接下来根据上面得到的返回进行更加细粒度的调节。
```
rf_param = {
    'n_estimators': np.arange(100, 200, 10),
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=50), param_grid=rf_param,
                         cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 10 candidates, totalling 30 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:   35.4s finished
    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;不出所料，这里的变异系数就存在明显的波动，而最终得分却没什么提升，说明`n_estimators`是一个适合在粗粒度上进行调节的参数，这里我们就选取验证变异系数最小的取值*160*作为模型中`n_estimators`参数的最终取值。
```
plot_acc_4_grid(rf_grid_cv, 'n_estimators')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_62_0.png)

* max_depth

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`max_depth`控制着每棵树的深度，随着树的深度越升，子模型的偏差降低而方差升高，当方差升高到一定程度的时候将会使泛化性能下降，也就是出现过拟合现象。如果是一颗完全二叉树，则最后叶节点将为$2^{n-1}$，这里使用的训练集的数量只需要一颗深度为*10*的树基本就能满足每个叶节点一个实例，最后的结果显然不会如此理想，所以现将深度的查找范围扩大到*100*进行训练。

```
rf_param = {
    'max_depth': np.arange(1, 100),
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160), param_grid=rf_param,
                         cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 99 candidates, totalling 297 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   31.3s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=-1)]: Done 297 out of 297 | elapsed:  5.5min finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_depth': array([ 1,  2, ..., 98, 99])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;下面让我们来看看这些模型的整体表现如何，通过第一张图可以清晰的看到随着深度的增加，模型在变得越来越好，但是我们并不能因此而沾沾自喜，和在训练`n_estimators`时一样，我们还要考察它的变异系数。前期随着模型得分越来越高，变异系数也在稳步下降，但是在`max_depth`增长到*5*的时候，得分的增幅明显放缓，到*9*之后更是基本保持不变了，而反观验证得分的变异系数，它也基本在同样的节点发生了逆转，在`max_depth`为*6*时达到波谷，而后便开始逐步上升，在*12*之后出现明显的波动，这说明得分开始趋于不稳定，调节这一部分取值并不会有多大好处。
```
plot_acc_4_grid(rf_grid_cv, 'max_depth')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_65_0.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在确定这个参数之前，为了更准确地估计，我把参数调节的范围缩小到了*1~20*，希望这能带来更清楚的认识。
```
rf_param = {
    'max_depth': np.arange(1, 20),
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160), param_grid=rf_param,
                         cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 19 candidates, totalling 57 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   28.9s
    [Parallel(n_jobs=-1)]: Done  57 out of  57 | elapsed:   44.3s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_depth': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;现在我们可以看到上面产生变化的那一部分的放大版了，现在我们可以明确地说明验证得分的变异系数发生转变是在`max_depth`为*5*的时候，在为*12*的时候开始出现波动。至于最后的取值，我们依然遵循前面的原则，分数尽可能高，而变异系数尽可能低，当然并不是说取对应的分数高和编译系数最低的，因为可能存在变异系数在低谷时得分并不高的情况。最后经过考虑之后暂时选择了$max_depth$为*5*。
```
plot_acc_4_grid(rf_grid_cv, 'max_depth')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_67_0.png)

* max_features

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`max_features`决定了算法在分裂节点时需要考虑的最大特征数，其可以通过内置的枚举值来计算个数也可以通过直接设置数值计算。因为这里的特征数量并不是很多，加上$OneHot$向量一共有345个，所以我采用了设置数值的方式，这样能够清晰地看到不同取值对模型的影响以及确定影响的具体数值。
```
rf_param = {
    'max_features': np.arange(1, 345, 10)
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 35 candidates, totalling 105 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    6.3s
    [Parallel(n_jobs=-1)]: Done 105 out of 105 | elapsed:   26.2s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_features': array([  1,  11,  21,  31,  41,  51,  61,  71,  81,  91, 101, 111, 121,
           131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251,
           261, 271, 281, 291, 301, 311, 321, 331, 341])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我想这幅图应该是目前为止最直观最简单的一副图了，其趋势非常明显——得分不断上升，变异系数不断下降，由此可知，只要`max_feature`设置为最大值就可以了。甚至还可以在特征工程中人工增加一些特征来提升模型的性能。
```
plot_acc_4_grid(rf_grid_cv, 'max_features')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_70_0.png)


* min_samples_split

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`min_samples_split`同样是一个影响模型偏差/方差的参数，它限定了一个节点需要分裂所需的最小样本数，其值越大模型越简单，偏差越大方差越小，而调节这个参数就是为了在这之间做一个权衡。`min_samples_split`参数最小取值为*2*，基于和`max_depth`一样的道理，这里将取值限定在*2~100*。

```
rf_param = {
    'min_samples_split': np.arange(2,100, 10),
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto'),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 10 candidates, totalling 30 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:   14.2s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_samples_split': array([ 2, 12, 22, 32, 42, 52, 62, 72, 82, 92])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这幅图有一个有趣的地方，虽然得分在不断地下降，但是变异系数存在一个低谷，如果训练到此为止，似乎有理由去选择这个值，因为其有不差的评分和相对较低的变异系数。但事实真的如此吗？
```
plot_acc_4_grid(rf_grid_cv, 'min_samples_split')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_73_0.png)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了验证上面的假设，我们将取值范围进一步缩小，同时将它的步长从*10*降低到*1*，让我们来看看训练效果如何。
```
rf_param = {
    'min_samples_split': np.arange(2,22),
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto'),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 20 candidates, totalling 60 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   21.7s
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:   28.4s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_samples_split': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 20, 21])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;没错，上面解释过的现象又再次出现了，在上面训练中的波谷并不是全局的波谷，而且可以明显地发现其振荡的现象，但其总体趋势是在升高。因此可以得出结论，`min_samples_split`在这里并不适合调节，只需要将其设置为默认值就行了。
```
plot_acc_4_grid(rf_grid_cv, 'min_samples_split')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_75_0.png)

* max_leaf_nodes

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`max_leaf_nodes`规定了最大叶节点数，与`min_samples_split`相反， `max_leaf_nodes`越大模型越复杂，方差越高。甚至可以不限制它的数量，任由其生长，基于此，我们这里选用一个较大的范围去观察它的整体趋势，然后再如同前面一样在细粒度上去进行调节。
```
rf_param = {
    'max_leaf_nodes': np.arange(2, 1000, 10)
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                                               min_samples_split=2),  
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 100 candidates, totalling 300 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   25.9s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  3.0min finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_leaf_nodes': array([  2,  12, ..., 982, 992])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;和前面大多数参数的表现一样，这里也是在取值较小时似乎就已经达到了比较不错的效果，后面的取值对结果也没有提升，反而徒增资源的消耗。
```
plot_acc_4_grid(rf_grid_cv, 'max_leaf_nodes')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_78_0.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;依葫芦画瓢地，我们缩小参数返回进行细粒度的调参，可以发现其在取值为*22*时表现已趋于稳定。
```
rf_param = {
    'max_leaf_nodes': np.arange(2, 200, 10)
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                                               min_samples_split=2),  
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 20 candidates, totalling 60 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   26.0s
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:   35.6s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_leaf_nodes': array([  2,  12,  22,  32,  42,  52,  62,  72,  82,  92, 102, 112, 122,
           132, 142, 152, 162, 172, 182, 192])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

```
plot_acc_4_grid(rf_grid_cv, 'max_leaf_nodes')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_80_0.png)

* min_weight_fraction_leaf

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;叶节点最小权重总值限制了叶子节点所有样本权重的最小值，如果小于这个值，则会和其他叶子节点一起被剪枝，提高模型偏差，降低方差。如果样本的分布存在偏斜或者有较多的缺失值可以考虑引入权重。由于之前已经在特征工程中处理了相应的问题，所以这里的调参对提升模型不会有什么作用，但是并不妨碍我们一窥究竟。
```
rf_param = {
    'min_weight_fraction_leaf': np.linspace(0, 0.5, 10)
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                                               min_samples_split=2, max_leaf_nodes=22 ),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 10 candidates, totalling 30 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    7.9s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=22,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_weight_fraction_leaf': array([0.     , 0.05556, 0.11111, 0.16667, 0.22222, 0.27778, 0.33333,
           0.38889, 0.44444, 0.5    ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我想这是这篇文章到此为止第二个如此直观的图了，那么我便不在做过多的解释，直接确定取值了。
```
plot_acc_4_grid(rf_grid_cv, 'min_weight_fraction_leaf')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_83_0.png)

* min_samples_leaf

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`min_samples_leaf`是我们训练的最后一个关于子模型结构的参数了，它表示叶节点的最小样本树。关于这个描述如果返回前面去看我们已经训练过的参数，会发现一个和它非常相似的参数，就是`min_samples_split`，这两个参树可以说是直接限定定了叶节点样本个数的范围。下面让我们仿照`min_samples_split`的训练过程对`min_samples_leaf`的参数进行设定。
```
rf_param = {
    'min_samples_leaf': np.arange(1, 100, 10)
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                                               min_samples_split=2, max_leaf_nodes=22, min_weight_fraction_leaf=0 ),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 10 candidates, totalling 30 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:   13.5s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=22,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_samples_leaf': array([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如图所示，虽然在较粗粒度的层面上进行调参，但是其总体趋势确实非常明显，所以这里便不再多做赘述，直接将值取为*1*。
```
plot_acc_4_grid(rf_grid_cv, 'min_samples_leaf')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_86_0.png)

* bootstrap

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;到目前为止，我们已经通过以上的步骤训练完了直接影响子模型结构的参数(除了`n_estimators`)，现在我们稍微站高一点，尝试一下对`booststrap`这个参数取不同的值，看看最后的效果如何。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`bootstrap`决定了是否对样本进行抽样，也就是说它对训练使用什么哪些样本起着至关重要的作用。一般而言，使用子采样会降低子模型之间的关联度，降低最终模型的方差，这也是**bagging**的做法。
```
rf_param = {
    'bootstrap': [True, False]
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                           min_samples_split=2, max_leaf_nodes=22, min_weight_fraction_leaf=0,
                          min_samples_leaf=1),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 2 candidates, totalling 6 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   6 out of   6 | elapsed:    4.4s remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done   6 out of   6 | elapsed:    4.4s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=22,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'bootstrap': [True, False]}, pre_dispatch='2*n_jobs',
           refit=True, return_train_score='warn', scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;经过验证，在这里展现出的结果也确实如此。那么我很有什么理由不使用默认值呢。
```
plot_acc_4_grid(rf_grid_cv, 'bootstrap')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_89_0.png)


* criterion

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在**scikit-learn**的*0.18*版中，`mae`作为新的计算方法被添加进来，以前都是使用`mse`来做为判断是否分裂节点的计算方法。既然如此我们也来尝试一下吧。
```
rf_param = {
    'criterion': ['mse', 'mae']
}
rf_grid_cv = GridSearchCV(RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                           min_samples_split=2, max_leaf_nodes=22, min_weight_fraction_leaf=0,
                          min_samples_leaf=1, bootstrap=True),
                          param_grid=rf_param, cv=3, verbose=True, n_jobs=-1)
rf_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 2 candidates, totalling 6 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   6 out of   6 | elapsed:   59.7s remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done   6 out of   6 | elapsed:   59.7s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=22,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0, n_estimators=160, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'criterion': ['mse', 'mae']}, pre_dispatch='2*n_jobs',
           refit=True, return_train_score='warn', scoring=None, verbose=True)

```
plot_acc_4_grid(rf_grid_cv, 'criterion')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_92_0.png)

* RandomForestRegressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;经过繁琐的步骤，我们终于可以着手训练一个完整的随机森林模型了。将上面的参数一一对应，直接使用训练数据划分验证测试。
```
rf = RandomForestRegressor(n_estimators=160, max_depth=5, max_features='auto',
                           min_samples_split=2, max_leaf_nodes=22, min_weight_fraction_leaf=0,
                          min_samples_leaf=1, bootstrap=True)
```
```
rf_pred = cross_val_predict(rf, x_train, y_train,
                                 verbose=True, n_jobs=-1, cv=3)
mse = mean_squared_error(y_train, rf_pred)
np.sqrt(mse)
```
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:    1.7s finished

    0.16115609724602975
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;就这样，我们最后得到了一个自己亲手调试的集成模型。对这最后的成果我们可以做一个简要的分析。首先相比之前我多次尝试训练的随机森林，它有着一个极大的改变，那就是它的训练曲线不在是一条接近*1*的水平线了，good job！！！这说明通过调参已经有效的缓解了严重的过拟合现象；其次，我们也可以发现一些问题，模型似乎仍然存在一定程度的过拟合(两条线靠得并不太近)，同时模型的准确率似乎有着明显的下降。
```
plot_learning_curve(rf, x_train, y_train)
```
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   20.1s finished
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_96_3.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;虽然最后的模型看起来并不是很完美的解决方案，但这至少可以作为一个里程碑，它证明了贪心策略的可行性同时产出了一个完整的集成模型。

## GBDT
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**梯度提升树**是一种*Boosting*方法，每个子模型拟合的是上一个子模型的预测结果与真实结果的残差，即先训练一个弱分类器，然后用这个弱分类器去预测数据集，得到的预测结果和正确的结果取差，然后将得到的残差作为数据集新的预测目标，下一个分类器再去你和这个残差，如此反复，最后将所有的弱分类器加权求和得到最终分类器，所以说梯度提升树是一种基于加法模型的算法。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GBDT通常采用高偏差低方差的基函数，一般是*CART Tree(分类回归树)*。因为是基于树的集成模型，那么它同样涉及到树的生成问题，例如深度、叶子节点个数、分隔所需最小样本树等等。基于此，对这部分参数的训练可以直接仿照*Random Forest*的训练过程，所以便不再占用大量的篇幅去描述。
* subsample

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相比于随机森林，GBDT多训练了一个叫做`subsample`的参数，这个参数的中文名叫做子采样率，它表示每一次训练弱分类器所使用的样本比例，如果$\lt 1.0$表示使用随机梯度提升，同时也能降低方差提高偏差。下面让我们看看它的训练效果。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们将值限定在一个极小数与*1*之间，然后取了*100*个值，查看它们的具体表现。
```
gbdt_param = {
    'subsample': np.linspace(1e-7, 1, 100),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2, max_leaf_nodes=12, min_weight_fraction_leaf=0,
                                                     min_samples_leaf=1),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 100 candidates, totalling 300 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  90 tasks      | elapsed:    8.1s
    [Parallel(n_jobs=-1)]: Done 240 tasks      | elapsed:   29.4s
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:   39.7s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=12,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sampl...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'subsample': array([1.00000e-07, 1.01011e-02, ..., 9.89899e-01, 1.00000e+00])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们可以发现随着`subsample`的取值越大，得分总体呈上升趋势并趋于平稳，变异系数整体呈下降趋势，然后开始震荡。
```
plot_acc_4_grid(gbdt_grid_cv, 'subsample')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_129_0.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;现在让我们把目光集中在急速下降的地方，放大它的内部表现。
```
gbdt_param = {
    'subsample': np.linspace(1e-7,2.22222e-01, 100),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2, max_leaf_nodes=12, min_weight_fraction_leaf=0,
                                                     min_samples_leaf=1),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```
    Fitting 3 folds for each of 100 candidates, totalling 300 fits

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done 136 tasks      | elapsed:    9.1s
    [Parallel(n_jobs=-1)]: Done 286 tasks      | elapsed:   24.6s
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:   25.8s finished

    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=12,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sampl...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'subsample': array([1.00000e-07, 2.24477e-03, ..., 2.19977e-01, 2.22222e-01])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;没错，和上面的趋势几乎如出一辙。这同时说明该参数不太适合在过小的粒度上进行调节。
```
plot_acc_4_grid(gbdt_grid_cv, 'subsample')
```
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_131_0.png)

* GBDT
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;没错，我们又得到了一个集成模型，现在让我们再看看这个模型的表现。

```
gbdt = GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2, max_leaf_nodes=12, min_weight_fraction_leaf=0,
                                                     min_samples_leaf=1, subsample=1)
```
```
gbdt_pred = cross_val_predict(gbdt, x_train, y_train)
mse = mean_squared_error(y_train, gbdt_pred)
np.sqrt(mse)
```
    0.1558966244053635
```
plot_learning_curve(gbdt, x_train, y_train)
```
    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    8.1s finished

![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_134_2.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相比于随机森林而言，它的均方误差变小了,同时准确率也有所提高。

### GBDT历史版本
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同时，这里将之前使用`RandomizedSearchCV`的版本罗列了出来，可以发现两者的巨大差异，虽然该版本的均方误差更低，但是其明显的过拟合现象就说明这不是一个好的模型了。
```
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

```
# gbdt_grid_cv.best_estimator_
```
```
# gbdt_grid_cv.best_score_
```
```
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

```
plot_learning_curve(gbdt, x_train, y_train)
```
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]

    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  4.5min finished
![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_139_3.png)


## XGBoost
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;现在，我们已经使用相同的方式训练了两个集成模型，当然，我还尝试了一些其他模型，比如这里的[**XGBoost**](https://xgboost.readthedocs.io/en/latest/)。如果我们还按照上面的行文方式，那么就会变成记流水账了。既然训练方法和过程都已经熟悉，那么这里便直接给出训练后的结果，转而介绍一下XGBoost吧。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*XGBoost*是在*GBDT*的基础上改进而来的一种*Boosting*算法，虽然是Boosting算法，但是其可以使用特征上的并行计算提升训练效率。它在训练之前会对数据进行排序，然后保存为*block*结构以便在后序地迭代中使用。同样因为该结构的存在，在节点分裂需要计算每个特征值的增益的时候，就可以多线程地进行。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*XGBoost*相比与*GBDT*还在代价函数中还加入了正则化方法用于控制模型的复杂度，这里正则化方法同样是降低模型的方差达到防止过拟合的目的。同时*XGBoost*不仅仅只是使用*CART*（`booster='gbtree'`）作为基分类器,还同时引入了线性分类器（`booster='gblinear'`）,当使用线性分类器时就如同带有*l1*或*l2*正则的逻辑斯蒂回归（分类）和线性回归（回归）。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同时，*XGBoost*还支持列抽样、自动学习缺失值的分裂方向等。*XGBoost*是一个非常优秀的模型，我这里只是大概做了一个梳理，想要完全掌握这个算法实在是还差得很远。具体的内容可以查看[官方文档](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)，也可一去翻看论文。
```
xg = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=0, reg_alpha=0.08163, colsample_bylevel=0.06122,
                                      colsample_bynode=0, colsample_bytree=1, min_child_weight=1)
xg_pred = cross_val_predict(xg, x_train, y_train,
                             verbose=True, n_jobs=-1, cv=5)
mse = mean_squared_error(y_train, xg_pred)
np.sqrt(mse)
```
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   12.1s finished

    0.1472574980184566

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最后我们来看看最终训练的模型的表现情况，可以发现它比之前的*GBDT*在结果上有不小的提升，两条曲线非常接近，也没有严重的过拟合问题，这可以说是一个不错的结果了。
```
plot_learning_curve(xg, x_train, y_train)
```
    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  1.1min finished

![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_175_2.png)


## stack
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在文章的开头我们对集成学习做了一个概述，其中省略了关于子模型结果结合的方法，现在便对其做一个补充。在对具体的结合策略进行表述之前，先来说是使用结合策略可以带来哪些好处：
1. 因为目标的假设空间很大，可能有多个假设在训练集上达到相同的性能，因此结合多个学习器可以减小单个学习器因误选而导致泛化性能不佳的风险
2. 由于局部极小值的存在，而单个学习器在有的局部极小值对应的泛化性能很差，多次训练学习器并对结果进行结合可以降低陷入糟糕局部极小值的风险
3. 目标的真是假设空间可能不在当前算法所考虑的假设空间中，多个学习器可以将相应的假设空间扩大从而摒除单个学习器无效的情况

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;综上所述，结合策略的使用可以提高泛化性能，扩大相应的假设空间，使得模型的性能变得更好。下面就让我们看看有哪些具体的策略吧。<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先最容易想到的便是针对回归的平均法和针对分类的投票法。其中平均法又分为简单平均和加权平均。所谓简单平均就是求所有学习器的总平均值，加权平均就是对每一个分类器分配一个权重($\sum ^{T}_{i=1}w_i = 1$)再求和，而简单平均是加权平均权重为$\frac{1}{T}$的特例。投票法
```
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


```
from sklearn.linear_model import  Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

ridge = KernelRidge(degree=2, alpha=0.05, kernel='polynomial')
lasso = Lasso(alpha=0.0005)
en = ElasticNet(max_iter=5000, selection='random')
gbdt = GradientBoostingRegressor(n_estimators=49, max_depth=5, max_features='auto',
                                                      min_samples_split=2, max_leaf_nodes=7, min_weight_fraction_leaf=0,
                                                     min_samples_leaf=1, subsample=1)

stack_models = StackModel(base_models=(ridge, gbdt, en), final_model=lasso)
stack_models.fit(x_train, y_train)
```




    StackModel(base_models=(KernelRidge(alpha=0.05, coef0=1, degree=2, gamma=None, kernel='polynomial',
          kernel_params=None), GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=5,
                 max_features='auto', max_leaf_nodes=7,
      ...False, precompute=False,
          random_state=None, selection='random', tol=0.0001, warm_start=False)),
          final_model=Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False),
          n_folds=5)




```
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
stack_pred = stack_models.predict(x_train)
mse = mean_squared_error(stack_pred, y_train)
np.sqrt(mse)
```




    0.08064604349182854




```
plot_learning_curve(stack_models, x_train, y_train)
```

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   49.9s finished



![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_186_2.png)


## LightGBM


```
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




```
lgb_grid_cv.best_score_
```




    0.9038911706779178




```
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




```
lgb_pred = cross_val_predict(lgb_grid_cv.best_estimator_, x_train, y_train,
                  cv=3, n_jobs=-1)
mse = mean_squared_error(y_train, lgb_pred)
np.sqrt(mse)
```




    0.12889382465830496




```
plot_learning_curve(lgb_grid_cv.best_estimator_, x_train, y_train)
```

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   11.1s finished



![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_192_2.png)


## 结合不同预测


```
ensemble_pred = stack_pred*0.8 + xg_pred*0.2
mse = mean_squared_error(ensemble_pred, y_train)
np.sqrt(mse)
```




    0.07707245327525736



# 预测


```
test = pd.read_csv('house_price/test.csv')
index = np.array(test[['Id']])[:,0]
test = test.set_index(['Id'])
x_test = full_pipeline.transform(test)
```


```
# ridge.fit(x_train, y_train)
pre =  stack_models.predict(x_test)
# xg_pre = xg.predict(x_test)
# ensemble_pred = stack_pre*0.8 + xg_pre*0.2
pred_df = pd.DataFrame({'Id':index,
                       'SalePrice':np.expm1(pre)})
pred_df.to_csv('./house_price/prediction.csv', index='')
```


```

```
[^1]: 又称离散系数，这里是标准差系数，其反应的是单位均值上的各指标观测值的离散程度。