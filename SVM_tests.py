import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Load in Data
df_hotel = pd.read_csv("new_hotel.csv")
print(df_hotel)

df_hotel.info()

df_hotel = df_hotel.head(2000)

# Instantiate X and y variables
X = df_hotel.drop("is_canceled",axis=1) #keep all feature columns (drop is_canceled)
y = df_hotel["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Define parameters: these will need to be tuned to prevent overfitting and underfitting
params = {
    "kernel": "linear",  # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', or 'precomputed'
    "C": 1,  # penalty parameter - tradeoff between smooth line and fitting the training points exactly. Regularization parameter, squared l2 penalty
    "gamma": 0.01, # the higher the gamma, the more it tries to exactly fit training data # Kernel coefficient (a float, 'scale', or 'auto') for 'rbf', 'poly' and 'sigmoid'
    "degree": 3, # when kernel set to 'poly'. the degree of the polynomial used. degree = 1 is same as linear kernel # Degree of ‘poly’ kernel function
    "random_state": 42,
}

# Create a svm.SVC with the parameters above
clf = svm.SVC(**params)

# Train the SVM classifer on the train set
clf = clf.fit(X_train, y_train)

# Predict the outcomes on the test set
y_pred = clf.predict(X_test)

print("Accuracy SVM:", accuracy_score(y_test, y_pred))

# Define a parameter grid with distributions of possible parameters to use
rs_param_grid = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [0.1, 1, 10],
    "gamma": [0.0001, 0.001, 0.01, 0.1],
}

# Create a svm.SVC object
clf = svm.SVC(random_state=42)

# Instantiate RandomizedSearchCV() with clf and the parameter grid
clf_rs = RandomizedSearchCV(
    estimator= clf, # object of svm.SVC type
    param_distributions= rs_param_grid, # dict of parameters to try
    cv=3,  # Number of folds
    n_iter=5,  # Number of parameter candidate settings to sample
    verbose=2,  # The higher this is, the more messages are displayed
    random_state=42,
)

# Train the model on the training set
clf_rs.fit(X_train, y_train)

# Print the best parameters and highest accuracy
print("Best SVM parameters found: ", clf_rs.best_params_)
print("Best SVM accuracy found: ", clf_rs.best_score_)
