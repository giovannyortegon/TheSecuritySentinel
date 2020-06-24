#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Logisticregresion
from sklearn.metrics import f1_store
from imblearn.over_sample import RandomOverSampler
from imblearn.over_sample import SMOTE

df = pd.read_csv('creditcard.csv')

y = df['Class']
x = df.drop('Class', axis=1)

print("Negative example before Oversample: ", len(y[y == 0]))
print("Positive example before Oversample: ", len(y[y == 1]))

# OverSample
X_res, y_res = SMOTE().fit_sample(X, y)

print("Negative example before Oversample: ", len(y[y == 0]))
print("Positive example before Oversample: ", len(y[y == 1]))

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                                    test_size=0.33,
                                                    random_state=17)

clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("F1 Score: ", f1_score(y_pred, y_test))
