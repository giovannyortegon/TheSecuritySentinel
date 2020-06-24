#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df = pd.read_csv('payment_fraud.csv')
df = pd.get_dummies(df, columns=['paymentMethod'])

Y = df['label']
X = df.drop(['label'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.33,
                                                    random_state=17)

clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("F1-Score: ", f1_score(Y_test, Y_pred))
