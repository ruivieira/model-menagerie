# Required Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Creating synthetic dataset with 5 features
X, y = make_classification(n_samples=10000, n_features=5, n_informative=3, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# Creating a DataFrame
df = pd.DataFrame(X, columns=['Age', 'Sex', 'Race', 'Salary', 'Location'])
df['Approved'] = y

# Create a bias in the data
df.loc[df['Sex'] > 0.2, 'Approved'] = 1
df.loc[df['Salary'] < 0.2, 'Approved'] = 0

# Splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Sex', 'Race', 'Salary', 'Location']], df['Approved'],
                                                    test_size=0.2, random_state=1)

# Building a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Serialise the model
path = "../artifacts/income.joblib"
print(f"Serialising the model to {path} ...")
joblib.dump(value=lr, filename=path)

# Predicting and getting accuracy
y_pred = lr.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")