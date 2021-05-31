import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

N_THREADS = 8
N_FOLDS = 5
RANDOM_STATE = 33
TEST_SIZE = 0.3
TIMEOUT = 3600 * 4

train = pd.read_csv("train.csv", header=0)
test = pd.read_csv("test.csv", header=0)

train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

X = train.iloc[:,:-2]
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
le = LabelEncoder()
train['target'] = le.fit_transform(train['target'])

automl = TabularUtilizedAutoML(task = Task('multiclass'),
                               timeout = TIMEOUT,
                               cpu_limit = N_THREADS,
                               verbose=0,
                               reader_params = {'n_jobs': N_THREADS}
)

target_column = 'target'
roles = {
    'target': target_column,
    'drop': ['id']
}

lightml_pred = automl.fit_predict(train, roles = roles)
#print('lightml_pred:\n{}\nShape = {}'.format(lightml_pred[:10], lightml_pred.shape))

test_pred = automl.predict(test)
#print('Prediction for test set:\n{}\nShape = {}'.format(test_pred[:5], test_pred.shape))
## 输出结果
auto_submission = pd.DataFrame(test_pred.data, columns=["Class_1", "Class_2", "Class_3", "Class_4"])

submission_df = pd.read_csv('sample_submission.csv')
submission_df["Class_1"] = auto_submission["Class_1"]
submission_df["Class_2"] = auto_submission["Class_2"]
submission_df["Class_3"] = auto_submission["Class_3"]
submission_df["Class_4"] = auto_submission["Class_4"]

submission_df.to_csv('Final_submission.csv', index=False)
