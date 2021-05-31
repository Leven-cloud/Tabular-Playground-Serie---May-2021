import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def multiclass_logloss(actual, predicted, eps=1e-15):
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

train = pd.read_csv("train.csv", header=0)
test = pd.read_csv("test.csv", header=0)

X = train.drop("target", axis=1)
#X = X.drop("feature_13", axis=1)
#X = X.drop("feature_36", axis=1)
y = train["target"].astype('category').cat.codes

#Class = pd.get_dummies(train["target"],prefix="target").head()

##使用XGBoost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

XB = XGBClassifier(objective='multi:softprob',
                            seed=42,
                            min_child_weight=2.3,
                            use_label_encoder=False,
                            num_class=4,
                            colsample_bytree=0.25,
                            subsample=0.9,
                            eta=0.011,
                            gamma = 0.25,
                            max_depth = 4,
                            reg_lambda = 8,
                            n_estimators = 500,
                            )

XB.fit(X_train, y_train)
XB_y_predict = XB.predict_proba(test)

# cv_params={'max_delta_step':[0,1]}
#
# other_params={
# 'base_score':0.3,
# 'colsample_bylevel':1,
# 'colsample_bytree':0.7,
# 'gamma':1,
# 'learning_rate':0.05,
# 'max_delta_step':0,
# 'max_depth':4,
# 'min_child_weight':4,
# 'n_estimators':85,
# 'reg_alpha':0.1,
# 'reg_lambda':0.05,
# 'subsample':0.7
# }
#
# model = xgb.XGBClassifier(**other_params)
# opt = GridSearchCV(model,cv_params,scoring='f1',cv=5)
# opt.fit(X_train, y_train)
# print(opt.best_score_)
# print(opt.best_params_)

for i in range(10):
    XB = XGBClassifier(objective='multi:softprob',
                            seed=42,
                            use_label_encoder=False,
                            num_class=4,
                            colsample_bytree=0.5,
                            subsample=0.9,
                            eta=0.3,
                            gamma = 0.25,
                            max_depth = 3,
                            reg_lambda = 10,
                            n_estimators = 500)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    XB.fit(X_train, y_train)
    XB_y_predict = (XB_y_predict + XB.predict_proba(test))/2.  #test  X_test
    #XB_y_predict = XB.predict_proba(test)

#mll = multiclass_logloss(y_test, XB_y_predict)

##输出结果
XB_submission = pd.DataFrame(XB_y_predict, columns=["Class_1", "Class_2", "Class_3", "Class_4"])

submission_df = pd.read_csv('sample_submission.csv')
submission_df["Class_1"] = XB_submission["Class_1"]
submission_df["Class_2"] = XB_submission["Class_2"]
submission_df["Class_3"] = XB_submission["Class_3"]
submission_df["Class_4"] = XB_submission["Class_4"]

# fig, ax = plt.subplots(figsize=(12, 14))
# xgb.plot_importance(XB, ax=ax)
# plt.show()
#
#print(mll)

submission_df.to_csv('XGB_submission.csv', index=False)
