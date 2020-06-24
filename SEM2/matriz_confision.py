#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time

df = pd.read_csv('payment_fraud.csv')
 
Y = df['label']
X = df.drop(['label', 'paymentMethod', 'accountAgeDays'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.33,
                                                    random_state=17)
clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)


print("\n", "Precision: ", accuracy_score(Y_pred, Y_test))
print("\n", "Tasa de error: ", 1 - accuracy_score(Y_pred, Y_test))
print("\n\n")
print(confusion_matrix(Y_test, Y_pred))
print("\n", "Precision: ", precision_score(Y_pred, Y_test))
