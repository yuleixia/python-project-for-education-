In [1]: import pandas as pd
   ...: import numpy as np
   ...: from sklearn.cross_validation import train_test_split
   ...: from sklearn.metrics import roc_auc_score
   ...: from sklearn.ensemble import ExtraTreesClassifier
   ...: from sklearn.feature_selection import SelectFromModel
   ...: import matplotlib.pyplot as plt
   ...: train = pd.read_csv("train.csv")
   ...: test = pd.read_csv("test.csv")
   ...: train = pd.read_csv("train.csv")
   ...: test = pd.read_csv("test.csv")
   ...: # clean and split data^M
   ...: # remove constant columns (std = 0)^M
   ...: remove = []
   ...: for col in train.columns:
   ...:     if train[col].std() == 0:
   ...:         remove.append(col)
   ...: train.drop(remove, axis=1, inplace=True)
   ...: test.drop(remove, axis=1, inplace=True)
In [3]: # remove duplicated columns^M
   ...: remove = []
   ...: cols = train.columns
   ...: for i in range(len(cols)-1):
   ...:     v = train[cols[i]].values
   ...:     for j in range(i+1,len(cols)):
   ...:         if np.array_equal(v,train[cols[j]].values):
   ...:             remove.append(cols[j])
   ...: train.drop(remove, axis=1, inplace=True)
   ...: test.drop(remove, axis=1, inplace=True)
   ...: # split data into train and test^M
   ...: test_id = test.ID
   ...: test = test.drop(["ID"],axis=1)
   ...: X = train.drop(["TARGET","ID"],axis=1)
   ...: y = train.TARGET.values
   ...: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
   ...: print(X_train.shape, X_test.shape, test.shape)
   ...: ## # Feature selection
   ...: clf = ExtraTreesClassifier(random_state=1729)
   ...: selector = clf.fit(X_train, y_train)
In [6]: feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)
   ...: feat_imp[:40].plot(kind='bar', title='Feature Importances according to ExtraTreesClassifier', figsize=(12, 8))
   ...: plt.ylabel('Feature Importance Score')
   ...: plt.subplots_adjust(bottom=0.3)
   ...: plt.savefig('1.png')
   ...: plt.show()
   ...: # clf.feature_importances_ 
   ...: fs = SelectFromModel(selector, prefit=True)
   ...: X_train = fs.transform(X_train)
   ...: X_test = fs.transform(X_test)
   ...: test = fs.transform(test)
In [7]: from sklearn.linear_model import LogisticRegression
In [8]: pivot = int(0.8 * len(X_train))
In [9]: X_train = np.array(X_train)
In [10]: y = np.array(y)
In [11]: validation_features = X_train[pivot:]
In [12]: y_train = np.array(y_train)
In [14]: y_test = np.array(y_test)
In [15]: X_test = np.array(X_test)
In [16]: print("Len validation:", len(X_test),"Len train:", len(X_train))
In [17]: model = LogisticRegression()
In [18]: model.fit(X_train, y_train)
In [19]: score = model.score(X_test, y_test)
In [20]: print(score)
In [21]: score_train = model.score(X_train, y_train)
In [22]: print(score_train)