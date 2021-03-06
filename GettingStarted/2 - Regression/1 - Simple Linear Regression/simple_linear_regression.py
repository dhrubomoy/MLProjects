import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fit simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set result
y_pred = regressor.predict(X_test)

# Visualize!
def visualize_results(X, y, title):
    plt.scatter(X, y, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='green')
    plt.title(title)
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()
    
visualize_results(X_train, y_train, title='Salary VS Experience (Training Set)')
visualize_results(X_test, y_test, title='Salary VS Experience (Test Set)')