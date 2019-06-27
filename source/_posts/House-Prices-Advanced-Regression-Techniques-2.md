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

# 集成学习


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

* n_estimators


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




```
plot_acc_4_grid(rf_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_60_0.png)



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




```
plot_acc_4_grid(rf_grid_cv, 'n_estimators')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_62_0.png)


* max_depth


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




```
plot_acc_4_grid(rf_grid_cv, 'max_depth')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_65_0.png)



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




```
plot_acc_4_grid(rf_grid_cv, 'max_depth')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_67_0.png)


* max_features


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




```
plot_acc_4_grid(rf_grid_cv, 'max_features')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_70_0.png)


* min_samples_split


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




```
plot_acc_4_grid(rf_grid_cv, 'min_samples_split')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_73_0.png)



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




```
plot_acc_4_grid(rf_grid_cv, 'min_samples_split')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_75_0.png)


* max_leaf_nodes


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




```
plot_acc_4_grid(rf_grid_cv, 'max_leaf_nodes')
```


![png](House-Prices-Advanced-Regression-Techniques-2/Predict%20House%20Prices_78_0.png)



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
