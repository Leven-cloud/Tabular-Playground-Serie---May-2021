import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

#Class = pd.get_dummies(train["target"],prefix="target").head()

##使用RandomForest

X = train.drop("target", axis=1)
y = train["target"].astype('category').cat.codes

rf = RandomForestClassifier(n_estimators=150,max_depth=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf.fit(X_train, y_train)
rf_y_predict = rf.predict_proba(test)  #test X_test

for i in range(10):
    rf = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf.fit(X_train, y_train)
    rf_y_predict = (rf_y_predict + rf.predict_proba(test))/2.  #test  X_test

#mll = multiclass_logloss(y_test, rf_y_predict)
#y_re = le.transform(int(rfr_y_predict))
#print(rf_y_predict)
# # 输出结果
rf_submission = pd.DataFrame(rf_y_predict, columns=["Class_1", "Class_2", "Class_3", "Class_4"])

submission_df = pd.read_csv('sample_submission.csv')
submission_df["Class_1"] = rf_submission["Class_1"]
submission_df["Class_2"] = rf_submission["Class_2"]
submission_df["Class_3"] = rf_submission["Class_3"]
submission_df["Class_4"] = rf_submission["Class_4"]

#print(mll)
submission_df.to_csv('RF_submission.csv', index=False)
