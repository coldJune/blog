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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;既然都单独将这部分内容拿出来描述了，那么在正式进入调参之前，我们先简要看看集成学习是个什么东西吧。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**集成学习**通过构建并结合多个学习器来完成学习任务，其先产生一组“个体学习器”，再用某种策略将它们结合起来。集成的方式又分为**同质集成**和**异质集成**。**同质集成**只包含相同类型的个体学习器，其个体学习器也称为“基学习器”；**异质集成**中的个体学习器是不同类型的，其被称为“组件学习器”。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;根据个体学习器的生成方式可以分为两大类，一种是串行生成的序列化方法，其个体学习器之间存在强依赖关系，代表为*Boosting*；另一种是同时生成的并行化方法，其个体学习器之间不存在强依赖关系，代表有*Bagging*和*Random Forest*。*Boosting*是一族可将弱学习器提升为强学习器的算法，它先从初始训练集训练出一个基学习器，再根据其表现调整训练样本的分布，即重新分配每个样本的权重，使表现不好的样本在下次训练时得到更多的关注，直至达到预定的训练次数，最后将所有的基学习器进行加权结合；*Boosting*每一次都是使用的全量数据，而*Bagging*却并不是，它采用有放回的采样的方式来生成训练集，每个基学习器使用不同的训练集来进行训练，有放回的采样使得同一个数据集能够被多次使用训练出不同的模型，最后可以通过投票(分类)和平均(回归)来结合各个基学习器的结果；*Random Forest*是在*Bagging*的基础上进一步在决策树的训练过程中引入随机属性选择,传统的决策树选择划分属性时是在当前节点的属性集合中选择一个最优属性，而在*RF*中是先从该结点的属性集合中随机选择一个包含$k$个属性的子集，再从子集中选择最优属性。
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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中的`n_estimators`的值一般来说是越大性能越好，泛化能力越强，是单调递增的，但随着子模型的数量增加，训练算法所消耗的资源和时间将会急剧增加。其它数值型参数对性能的影响都呈现出有增有减的，而枚举型的例如`criterion`则需要视情况而定了。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们已经对需要调节的参数有了一个直观的认识，知道每个参数代表的含义。但是我们怎么来调节他们呢？在一开始的时候，我选择了一种非常笨重的方式，我通过直接将所有参数塞进`GridSearchCV`，而这导致训练需要花费大量的时间，不如设想有*3*个参数需要调节，每个参数取*10*个待定值，最后需要尝试的组合高达**1000**个之多，而这里的参数有`9`个之多，如果是更复杂的神经网络，那基本上就是望山跑死马的事了。我后知后觉得意识到网格查找的局限性，又想起书里提到的随机方法`RandomizedSearchCV`，我马上进行了尝试，并一度以为这样就能完美解决问题，但显然我是过于乐观了——最后训练出的模型基本上都是过拟合的，而且参数可控性极低。
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




```
plot_acc_4_grid(rf_grid_cv, 'min_weight_fraction_leaf')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_83_0.png)


* min_samples_leaf


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




```
plot_acc_4_grid(rf_grid_cv, 'min_samples_leaf')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_86_0.png)


* bootstrap


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




```
plot_acc_4_grid(rf_grid_cv, 'bootstrap')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_89_0.png)


* criterion


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




```
plot_learning_curve(rf, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   20.1s finished



![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_96_3.png)


## GBDT

* n_estimators


```
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
gbdt_param = {
    'n_estimators': np.arange(1, 3000, 100),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 30 candidates, totalling 90 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:  6.8min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_sampl...=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([   1,  101,  201,  301,  401,  501,  601,  701,  801,  901, 1001,
           1101, 1201, 1301, 1401, 1501, 1601, 1701, 1801, 1901, 2001, 2101,
           2201, 2301, 2401, 2501, 2601, 2701, 2801, 2901])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_100_0.png)



```
gbdt_param = {
    'n_estimators': np.arange(1, 200),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 199 candidates, totalling 597 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    3.0s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:   24.4s
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=-1)]: Done 597 out of 597 | elapsed:  3.3min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_sampl...=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([  1,   2, ..., 198, 199])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_102_0.png)



```
gbdt_param = {
    'n_estimators': np.arange(10, 70),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 60 candidates, totalling 180 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    3.7s
    [Parallel(n_jobs=-1)]: Done 180 out of 180 | elapsed:   25.3s finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_sampl...=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
           44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
           61, 62, 63, 64, 65, 66, 67, 68, 69])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_104_0.png)


* max_depth


```
gbdt_param = {
    'max_depth': np.arange(1, 100),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 99 candidates, totalling 297 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  55 tasks      | elapsed:   32.4s
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=-1)]: Done 297 out of 297 | elapsed:  4.7min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_sampl...=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_depth': array([ 1,  2, ..., 98, 99])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'max_depth')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_107_0.png)


* max_features


```
gbdt_param = {
    'max_features': np.arange(1, 345, 10),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 35 candidates, totalling 105 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  88 tasks      | elapsed:   19.7s
    [Parallel(n_jobs=-1)]: Done 105 out of 105 | elapsed:   26.2s finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_sampl...=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_features': array([  1,  11,  21,  31,  41,  51,  61,  71,  81,  91, 101, 111, 121,
           131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251,
           261, 271, 281, 291, 301, 311, 321, 331, 341])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'max_features')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_110_0.png)



```
gbdt_param = {
    'max_features': np.arange(61, 81),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 20 candidates, totalling 60 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    5.6s
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:    7.6s finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_sampl...=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_features': array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
           78, 79, 80])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'max_features')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_112_0.png)


* min_samples_split


```
gbdt_param = {
    'min_samples_split': np.arange(2,100),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto'),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 98 candidates, totalling 294 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   15.1s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:   49.9s
    [Parallel(n_jobs=-1)]: Done 294 out of 294 | elapsed:  1.2min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sam...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_samples_split': array([ 2,  3, ..., 98, 99])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'min_samples_split')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_115_0.png)


* max_leaf_nodes


```
gbdt_param = {
    'max_leaf_nodes': np.arange(2,1000, 10),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   27.4s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  4.7min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sam...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_leaf_nodes': array([  2,  12, ..., 982, 992])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'max_leaf_nodes')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_118_0.png)



```
gbdt_param = {
    'max_leaf_nodes': np.arange(2,100),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 98 candidates, totalling 294 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    7.0s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 294 out of 294 | elapsed:  2.4min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sam...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_leaf_nodes': array([ 2,  3, ..., 98, 99])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'max_leaf_nodes')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_120_0.png)


* min_weight_fraction_leaf


```
gbdt_param = {
    'min_weight_fraction_leaf': np.linspace(0,0.5, 10),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2, max_leaf_nodes=12),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    2.4s finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=12,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sampl...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_weight_fraction_leaf': array([0.     , 0.05556, 0.11111, 0.16667, 0.22222, 0.27778, 0.33333,
           0.38889, 0.44444, 0.5    ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'min_weight_fraction_leaf')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_123_0.png)


* min_samples_leaf


```
gbdt_param = {
    'min_samples_leaf': np.arange(1,100, 10),
}
gbdt_grid_cv = GridSearchCV(GradientBoostingRegressor(n_estimators=23, max_depth=8, max_features='auto',
                                                      min_samples_split=2, max_leaf_nodes=12, min_weight_fraction_leaf=0),
                            param_grid=gbdt_param, verbose=True, cv=3, n_jobs=-1)
gbdt_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    5.0s finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=8,
                 max_features='auto', max_leaf_nodes=12,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_sampl...       subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_samples_leaf': array([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(gbdt_grid_cv, 'min_samples_leaf')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_126_0.png)


* subsample


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




```
plot_acc_4_grid(gbdt_grid_cv, 'subsample')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_129_0.png)



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




```
plot_acc_4_grid(gbdt_grid_cv, 'subsample')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_131_0.png)



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

* n_estimators


```
from xgboost import XGBRegressor
xg_param = {
    'n_estimators': np.arange(100, 1000,100),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree'),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 9 candidates, totalling 27 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  27 out of  27 | elapsed:  1.2min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([100, 200, 300, 400, 500, 600, 700, 800, 900])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_143_0.png)



```
xg_param = {
    'n_estimators': np.arange(100, 2000, 100),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree'),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 19 candidates, totalling 57 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  4.2min
    [Parallel(n_jobs=-1)]: Done  57 out of  57 | elapsed:  7.3min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([ 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000, 1100,
           1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_145_0.png)



```
xg_param = {
    'n_estimators': np.arange(450, 600, 10),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree'),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed:  3.6min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570,
           580, 590])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_147_0.png)



```
xg_param = {
    'n_estimators': np.arange(460, 480),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree'),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 20 candidates, totalling 60 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  3.2min
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:  4.5min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'n_estimators': array([460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472,
           473, 474, 475, 476, 477, 478, 479])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_149_0.png)



```
xg_grid_cv.cv_results_
```




    {'mean_fit_time': array([16.77087108, 16.56628791, 16.53286195, 16.48052835, 18.50080125,
            17.88966179, 17.50711584, 16.84727049, 16.20652898, 17.28450465,
            17.48412037, 17.0510006 , 16.60813022, 18.08475184, 19.80619462,
            21.77392968, 18.87454247, 18.52810876, 18.63795424, 18.59706275]),
     'std_fit_time': array([0.06452758, 0.31981148, 0.13711394, 0.14862619, 0.19553549,
            0.75645545, 0.38300823, 0.39934476, 0.20871311, 0.51859026,
            0.53054543, 0.11082421, 0.30885861, 0.70884986, 1.37727965,
            0.08492364, 0.06974826, 0.11424264, 0.5682274 , 0.48720384]),
     'mean_score_time': array([0.01766666, 0.02504349, 0.01392539, 0.01975203, 0.01903812,
            0.01699042, 0.02690125, 0.01732858, 0.01528366, 0.01802301,
            0.01720095, 0.01680136, 0.02672577, 0.0214026 , 0.02584465,
            0.01834901, 0.02193697, 0.02316133, 0.0175554 , 0.01474897]),
     'std_score_time': array([0.00455945, 0.01269679, 0.00109658, 0.00328442, 0.00749728,
            0.00206802, 0.00671113, 0.00277125, 0.00363458, 0.00418345,
            0.00255097, 0.00276758, 0.00976858, 0.00425326, 0.00530614,
            0.00140423, 0.00603777, 0.00647161, 0.00263934, 0.00327119]),
     'param_n_estimators': masked_array(data=[460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470,
                        471, 472, 473, 474, 475, 476, 477, 478, 479],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'n_estimators': 460},
      {'n_estimators': 461},
      {'n_estimators': 462},
      {'n_estimators': 463},
      {'n_estimators': 464},
      {'n_estimators': 465},
      {'n_estimators': 466},
      {'n_estimators': 467},
      {'n_estimators': 468},
      {'n_estimators': 469},
      {'n_estimators': 470},
      {'n_estimators': 471},
      {'n_estimators': 472},
      {'n_estimators': 473},
      {'n_estimators': 474},
      {'n_estimators': 475},
      {'n_estimators': 476},
      {'n_estimators': 477},
      {'n_estimators': 478},
      {'n_estimators': 479}],
     'split0_test_score': array([0.91093677, 0.91093459, 0.91095287, 0.91096842, 0.91096134,
            0.91100932, 0.91096652, 0.91082328, 0.91079769, 0.91079868,
            0.91083626, 0.91089773, 0.91089552, 0.91089421, 0.91092657,
            0.91089069, 0.91087079, 0.91090466, 0.91091567, 0.9109092 ]),
     'split1_test_score': array([0.89570281, 0.89568197, 0.89568796, 0.8956492 , 0.89558205,
            0.89548694, 0.89546541, 0.89555158, 0.89554495, 0.89556411,
            0.89562573, 0.89566419, 0.89566833, 0.8956966 , 0.89564484,
            0.89563801, 0.8956351 , 0.89564928, 0.89565577, 0.8956636 ]),
     'split2_test_score': array([0.90134347, 0.90134975, 0.90130296, 0.90127834, 0.90128112,
            0.90129138, 0.90130003, 0.90124756, 0.90123981, 0.90122449,
            0.90131101, 0.90124636, 0.9011988 , 0.90124558, 0.9012511 ,
            0.90121633, 0.90119993, 0.90116283, 0.90122346, 0.90124819]),
     'mean_test_score': array([0.90266102, 0.90265543, 0.90264793, 0.90263198, 0.90260817,
            0.90259588, 0.90257732, 0.9025408 , 0.90252748, 0.90252909,
            0.902591  , 0.90260276, 0.90258755, 0.90261213, 0.9026075 ,
            0.90258168, 0.90256861, 0.90257226, 0.9025983 , 0.902607  ]),
     'std_test_score': array([0.00628863, 0.00629493, 0.00630402, 0.00632687, 0.00634831,
            0.00640377, 0.00639243, 0.00630135, 0.00629313, 0.00628753,
            0.00627529, 0.00629259, 0.00629356, 0.0062792 , 0.00631204,
            0.00630128, 0.00629479, 0.00630722, 0.00630522, 0.00629772]),
     'rank_test_score': array([ 1,  2,  3,  4,  6, 11, 15, 18, 20, 19, 12,  9, 13,  5,  7, 14, 17,
            16, 10,  8], dtype=int32),
     'split0_train_score': array([0.99066379, 0.99069611, 0.99073952, 0.99076018, 0.99077522,
            0.99081363, 0.99085459, 0.99092424, 0.99097586, 0.99099333,
            0.99104144, 0.99106673, 0.99107897, 0.9911102 , 0.9911236 ,
            0.99115969, 0.99118241, 0.99122721, 0.99124871, 0.99126094]),
     'split1_train_score': array([0.99144731, 0.99150659, 0.99153146, 0.99154989, 0.99159965,
            0.9916188 , 0.99162954, 0.9916871 , 0.99170323, 0.99173422,
            0.99178526, 0.99183032, 0.99185063, 0.99186357, 0.99189252,
            0.99191176, 0.99193145, 0.99195966, 0.99197622, 0.99199422]),
     'split2_train_score': array([0.99184967, 0.99188183, 0.99190751, 0.99191864, 0.99193368,
            0.9919685 , 0.99198448, 0.99200409, 0.99202025, 0.99207444,
            0.9920818 , 0.99210266, 0.99212737, 0.992164  , 0.99219503,
            0.99222795, 0.99228846, 0.99230885, 0.99234229, 0.9923551 ]),
     'mean_train_score': array([0.99132026, 0.99136151, 0.99139283, 0.99140957, 0.99143618,
            0.99146698, 0.99148954, 0.99153848, 0.99156645, 0.99160066,
            0.99163616, 0.99166657, 0.99168566, 0.99171259, 0.99173705,
            0.99176647, 0.99180077, 0.99183191, 0.99185574, 0.99187009]),
     'std_train_score': array([0.0004924 , 0.00049482, 0.0004868 , 0.00048324, 0.00048686,
            0.00048354, 0.00047178, 0.0004532 , 0.0004372 , 0.00045135,
            0.00043761, 0.00043848, 0.00044362, 0.00044326, 0.00045101,
            0.00044805, 0.0004609 , 0.00045072, 0.00045451, 0.00045523])}



* max_depth


```
xg_param = {
    'max_depth': np.arange(1, 100),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 99 candidates, totalling 297 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  4.5min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed: 23.7min
    [Parallel(n_jobs=-1)]: Done 297 out of 297 | elapsed: 36.9min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'max_depth': array([ 1,  2, ..., 98, 99])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'max_depth')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_153_0.png)


* reg_lambda


```
xg_param = {
    'reg_lambda': np.linspace(0, 1),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 50 candidates, totalling 150 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.7min
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:  8.8min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'reg_lambda': array([0.     , 0.02041, 0.04082, 0.06122, 0.08163, 0.10204, 0.12245,
           0.14286, 0.16327, 0.18367, 0.20408, 0.22449, 0.2449 , 0.26531,
           0.28571, 0.30612, 0.32653, 0.34694, 0.36735, 0.38776, 0.40816,
           0.42857, 0.44898, 0.46939, 0.4898 , 0.5102 , 0.53061, 0.5...33, 0.83673,
           0.85714, 0.87755, 0.89796, 0.91837, 0.93878, 0.95918, 0.97959,
           1.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'reg_lambda')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_156_0.png)


* reg_alpha


```
xg_param = {
    'reg_alpha': np.linspace(0, 1),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=1),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 50 candidates, totalling 150 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.4min
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed: 11.7min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'reg_alpha': array([0.     , 0.02041, 0.04082, 0.06122, 0.08163, 0.10204, 0.12245,
           0.14286, 0.16327, 0.18367, 0.20408, 0.22449, 0.2449 , 0.26531,
           0.28571, 0.30612, 0.32653, 0.34694, 0.36735, 0.38776, 0.40816,
           0.42857, 0.44898, 0.46939, 0.4898 , 0.5102 , 0.53061, 0.55...33, 0.83673,
           0.85714, 0.87755, 0.89796, 0.91837, 0.93878, 0.95918, 0.97959,
           1.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'reg_alpha')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_159_0.png)


* colsample_bylevel


```
xg_param = {
    'colsample_bylevel': np.linspace(0, 1),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=0, reg_alpha=0.08163),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 50 candidates, totalling 150 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:  7.0min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0.08163, reg_lambda=0, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'colsample_bylevel': array([0.     , 0.02041, 0.04082, 0.06122, 0.08163, 0.10204, 0.12245,
           0.14286, 0.16327, 0.18367, 0.20408, 0.22449, 0.2449 , 0.26531,
           0.28571, 0.30612, 0.32653, 0.34694, 0.36735, 0.38776, 0.40816,
           0.42857, 0.44898, 0.46939, 0.4898 , 0.5102 , 0.530...33, 0.83673,
           0.85714, 0.87755, 0.89796, 0.91837, 0.93878, 0.95918, 0.97959,
           1.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'colsample_bylevel')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_162_0.png)


* colsample_bynode


```
xg_param = {
    'colsample_bynode': np.linspace(0, 1),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=0, reg_alpha=0.08163, colsample_bylevel=0.06122),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 50 candidates, totalling 150 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:  4.6min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.06122,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0.08163, reg_lambda=0, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'colsample_bynode': array([0.     , 0.02041, 0.04082, 0.06122, 0.08163, 0.10204, 0.12245,
           0.14286, 0.16327, 0.18367, 0.20408, 0.22449, 0.2449 , 0.26531,
           0.28571, 0.30612, 0.32653, 0.34694, 0.36735, 0.38776, 0.40816,
           0.42857, 0.44898, 0.46939, 0.4898 , 0.5102 , 0.5306...33, 0.83673,
           0.85714, 0.87755, 0.89796, 0.91837, 0.93878, 0.95918, 0.97959,
           1.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'colsample_bynode')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_165_0.png)



```
xg_param = {
    'colsample_bynode': np.linspace(0.91837, 1),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=0, reg_alpha=0.08163, colsample_bylevel=0.06122),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 50 candidates, totalling 150 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:  5.4min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.06122,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0.08163, reg_lambda=0, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'colsample_bynode': array([0.91837, 0.92004, 0.9217 , 0.92337, 0.92503, 0.9267 , 0.92837,
           0.93003, 0.9317 , 0.93336, 0.93503, 0.9367 , 0.93836, 0.94003,
           0.94169, 0.94336, 0.94502, 0.94669, 0.94836, 0.95002, 0.95169,
           0.95335, 0.95502, 0.95669, 0.95835, 0.96002, 0.9616...01, 0.98667,
           0.98834, 0.99   , 0.99167, 0.99334, 0.995  , 0.99667, 0.99833,
           1.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'colsample_bynode')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_167_0.png)


* colsample_bytree


```
xg_param = {
    'colsample_bytree': np.linspace(0, 1),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=0, reg_alpha=0.08163, colsample_bylevel=0.06122,
                                      colsample_bynode=0),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 50 candidates, totalling 150 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:  4.3min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.06122,
           colsample_bynode=0, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0.08163, reg_lambda=0, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'colsample_bytree': array([0.     , 0.02041, 0.04082, 0.06122, 0.08163, 0.10204, 0.12245,
           0.14286, 0.16327, 0.18367, 0.20408, 0.22449, 0.2449 , 0.26531,
           0.28571, 0.30612, 0.32653, 0.34694, 0.36735, 0.38776, 0.40816,
           0.42857, 0.44898, 0.46939, 0.4898 , 0.5102 , 0.5306...33, 0.83673,
           0.85714, 0.87755, 0.89796, 0.91837, 0.93878, 0.95918, 0.97959,
           1.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'colsample_bytree')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_170_0.png)


* min_child_weight


```
xg_param = {
    'min_child_weight': np.arange(1, 100),
}
xg_grid_cv = GridSearchCV(XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=470,
                                       max_depth=2, reg_lambda=0, reg_alpha=0.08163, colsample_bylevel=0.06122,
                                      colsample_bynode=0, colsample_bytree=1),
                            param_grid=xg_param, verbose=True, cv=3, n_jobs=-1)
xg_grid_cv.fit(x_train, y_train)
```

    Fitting 3 folds for each of 99 candidates, totalling 297 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  5.7min
    [Parallel(n_jobs=-1)]: Done 297 out of 297 | elapsed:  8.8min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.06122,
           colsample_bynode=0, colsample_bytree=1, gamma=0,
           importance_type='gain', learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=470,
           n_jobs=1, nthread=None, objective='reg:squarederror',
           random_state=0, reg_alpha=0.08163, reg_lambda=0, scale_pos_weight=1,
           seed=None, silent=None, subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'min_child_weight': array([ 1,  2, ..., 98, 99])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=True)




```
plot_acc_4_grid(xg_grid_cv, 'min_child_weight')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_173_0.png)



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




```
plot_learning_curve(xg, x_train, y_train)
```

    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  1.1min finished



![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_175_2.png)



```
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




```
#  xg_grid_cv.best_estimator_
```


```
# xg_grid_cv.best_score_
```


```
xg_pred = cross_val_predict(xg, x_train, y_train,
                            cv=3, verbose=True, n_jobs=-1)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:  1.5min finished



```
mse = mean_squared_error(y_train, xg_pred)
np.sqrt(mse)
```




    0.12419601284479917




```
plot_learning_curve(xg, x_train, y_train)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.


    [learning_curve] Training set sizes: [ 131  426  721 1016 1312]


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 21.1min finished



![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_181_3.png)


## stack


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