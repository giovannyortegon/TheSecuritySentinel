#!/usr/bin/env python3

from feature_selector import FeatureSelector
import pandas as pd

df = pd.read_csv('payment_fraud.csv')
#df = pd.get_dummies(df, columns=['paymentMethod'])

Y = df['label']
#X = df.drop('label', axis=1)
X = df.drop(['label', 'paymentMethod'], axis=1)

#fs = FeatureSelector(data = X, labels = Y)
# valores perdidos
#fs.identify_missing(missing_threshold=0.8)
#print(fs.missing_stats.head(10))

# valores unicos
#fs.identify_single_unique()
#print(fs.unique_stats.sample(7))

# caracteristicas colineales (altamente correlacionales)
#fs.identify_collinear(correlation_threshold=0.8)
#print(fs.record_collinear.head())

# caracteristicas con cero importancia
#fs.identify_zero_importance(task='classification', eval_metric = 'auc',
                            n_iterations = 10, early_stopping = False)
#fs.plot_feature_importances(threshold=0.99, plot_n = 12)
#print(fs.feature_importances.head(10))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.33,
                                                    random_state=17)

clf = Lo
