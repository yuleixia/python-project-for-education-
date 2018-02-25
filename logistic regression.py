In [1]: import pandas as pd^M
   ...: import numpy as np^M
   ...: ^M
   ...: from sklearn.cross_validation import train_test_split^M
   ...: from sklearn.metrics import roc_auc_score^M
   ...: from sklearn.ensemble import ExtraTreesClassifier^M
   ...: from sklearn.feature_selection import SelectFromModel^M
   ...: ^M
   ...: import xgboost as xgb^M
   ...: import matplotlib.pyplot as plt^M
   ...: ^M
   ...: train = pd.read_csv("train_encoded.csv")^M
   ...: test = pd.read_csv("test_encoded.csv")^M
   ...: ^M
   ...: # clean and split data^M
   ...: ^M
   ...: # remove constant columns (std = 0)^M
   ...: remove = []^M
   ...: for col in train.columns:^M
   ...:     if train[col].std() == 0:^M
   ...:         remove.append(col)^M
   ...: ^M
   ...: train.drop(remove, axis=1, inplace=True)^M
   ...: test.drop(remove, axis=1, inplace=True)
D:\Software\python3.6\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-1-a0c1eeb1dceb> in <module>()
      7 from sklearn.feature_selection import SelectFromModel
      8
----> 9 import xgboost as xgb
     10 import matplotlib.pyplot as plt
     11

ModuleNotFoundError: No module named 'xgboost'

In [2]: import pandas as pd^M
   ...: import numpy as np^M
   ...: ^M
   ...: from sklearn.cross_validation import train_test_split^M
   ...: from sklearn.metrics import roc_auc_score^M
   ...: from sklearn.ensemble import ExtraTreesClassifier^M
   ...: from sklearn.feature_selection import SelectFromModel^M
   ...: import matplotlib.pyplot as plt^M
   ...: ^M
   ...: train = pd.read_csv("train_encoded.csv")^M
   ...: test = pd.read_csv("test_encoded.csv")^M
   ...: ^M
   ...: # clean and split data^M
   ...: ^M
   ...: # remove constant columns (std = 0)^M
   ...: remove = []^M
   ...: for col in train.columns:^M
   ...:     if train[col].std() == 0:^M
   ...:         remove.append(col)^M
   ...: ^M
   ...: train.drop(remove, axis=1, inplace=True)^M
   ...: test.drop(remove, axis=1, inplace=True)

In [3]: # remove duplicated columns^M
   ...: remove = []^M
   ...: cols = train.columns^M
   ...: for i in range(len(cols)-1):^M
   ...:     v = train[cols[i]].values^M
   ...:     for j in range(i+1,len(cols)):^M
   ...:         if np.array_equal(v,train[cols[j]].values):^M
   ...:             remove.append(cols[j])^M
   ...: ^M
   ...: train.drop(remove, axis=1, inplace=True)^M
   ...: test.drop(remove, axis=1, inplace=True)^M
   ...: ^M
   ...: # split data into train and test^M
   ...: test_id = test.ID^M
   ...: test = test.drop(["ID"],axis=1)^M
   ...: ^M
   ...: X = train.drop(["TARGET","ID"],axis=1)^M
   ...: y = train.TARGET.values^M
   ...: ^M
   ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)^M
   ...: print(X_train.shape, X_test.shape, test.shape)^M
   ...: ^M
   ...: ## # Feature selection^M
   ...: clf = ExtraTreesClassifier(random_state=1729)^M
   ...: selector = clf.fit(X_train, y_train)
(60816, 306) (15204, 306) (75818, 306)

In [4]: X_train.type()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-4-10ef4c0c8447> in <module>()
----> 1 X_train.type()

D:\Software\python3.6\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
   3079             if name in self._info_axis:
   3080                 return self[name]
-> 3081             return object.__getattribute__(self, name)
   3082
   3083     def __setattr__(self, name, value):

AttributeError: 'DataFrame' object has no attribute 'type'

In [5]: X_train.head()
Out[5]:
       var3  var15  imp_ent_var16_ult1  imp_op_var39_comer_ult1  \
36768     2     33                 0.0                   590.16
26943    12     49                 0.0                     0.00
7908      2     23                 0.0                    80.40
58672     2     23                 0.0                     0.00
35728     2     43                 0.0                     0.00

       imp_op_var39_comer_ult3  imp_op_var40_comer_ult1  \
36768                   590.16                      0.0
26943                     0.00                      0.0
7908                    160.38                      0.0
58672                     0.00                      0.0
35728                     0.00                      0.0

       imp_op_var40_comer_ult3  imp_op_var40_efect_ult1  \
36768                      0.0                      0.0
26943                      0.0                      0.0
7908                       0.0                      0.0
58672                      0.0                      0.0
35728                      0.0                      0.0

       imp_op_var40_efect_ult3  imp_op_var40_ult1      ...        \
36768                      0.0                0.0      ...
26943                      0.0                0.0      ...
7908                       0.0                0.0      ...
58672                      0.0                0.0      ...
35728                      0.0                0.0      ...

       saldo_medio_var29_ult3  saldo_medio_var33_hace2  \
36768                     0.0                      0.0
26943                     0.0                      0.0
7908                      0.0                      0.0
58672                     0.0                      0.0
35728                     0.0                      0.0

       saldo_medio_var33_hace3  saldo_medio_var33_ult1  \
36768                      0.0                     0.0
26943                      0.0                     0.0
7908                       0.0                     0.0
58672                      0.0                     0.0
35728                      0.0                     0.0

       saldo_medio_var33_ult3  saldo_medio_var44_hace2  \
36768                     0.0                      0.0
26943                     0.0                      0.0
7908                      0.0                      0.0
58672                     0.0                      0.0
35728                     0.0                      0.0

       saldo_medio_var44_hace3  saldo_medio_var44_ult1  \
36768                      0.0                     0.0
26943                      0.0                     0.0
7908                       0.0                     0.0
58672                      0.0                     0.0
35728                      0.0                     0.0

       saldo_medio_var44_ult3          var38
36768                     0.0  117310.979016
26943                     0.0  117310.979016
7908                      0.0  121093.650000
58672                     0.0   69751.200000
35728                     0.0   32639.610000

[5 rows x 306 columns]

In [6]: feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)^M
   ...: feat_imp[:40].plot(kind='bar', title='Feature Importances according to ExtraTreesClassifier', figsize=(12, 8))
   ...: ^M
   ...: plt.ylabel('Feature Importance Score')^M
   ...: plt.subplots_adjust(bottom=0.3)^M
   ...: plt.savefig('1.png')^M
   ...: plt.show()^M
   ...: ^M
   ...: # clf.feature_importances_ ^M
   ...: fs = SelectFromModel(selector, prefit=True)^M
   ...: ^M
   ...: X_train = fs.transform(X_train)^M
   ...: X_test = fs.transform(X_test)^M
   ...: test = fs.transform(test)^M
   ...: ^M
(60816, 36) (15204, 36) (75818, 36)

In [7]: from sklearn.linear_model import LogisticRegression

In [8]: pivot = int(0.8 * len(X_train))

In [9]: X_train = np.array(X_train)

In [10]: y = np.array(y)

In [11]: validation_features = X_train[pivot:]

In [12]: y_train = np.array(y_train)

In [13]: y_test = np.arrat(y_test)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-13-6f4941185426> in <module>()
----> 1 y_test = np.arrat(y_test)

AttributeError: module 'numpy' has no attribute 'arrat'

In [14]: y_test = np.array(y_test)

In [15]: X_test = np.array(X_test)

In [16]: print("Len validation:", len(X_test),"Len train:", len(X_train))
Len validation: 15204 Len train: 60816

In [17]: model = LogisticRegression()

In [18]: model.fit(X_train, y_train)
Out[18]:
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

In [19]: score = model.score(X_test, y_test)

In [20]: print(score)
0.958958168903

In [21]: score_train = model.score(X_train, y_train)

In [22]: print(score_train)
0.960701131281

In [23]: