#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale

df = pd.read_csv('dataset.csv')

Y = df['Class']
X = df.drop(['Class'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.33,
                                                    random_state=17)

print(X_train.head(1))
print(normalize(X_train)[:1,:])
print(scale(X_train)[:1,:])
