import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#==============================================================================
# # take care of missing data
# imputer = Imputer()
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
#==============================================================================

#==============================================================================
# # encode catagorical data
# labelEncoder_X = LabelEncoder()
# X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
# ohe = OneHotEncoder(categorical_features=[0])
# X = ohe.fit_transform(X).toarray()
# labelEncoder_y = LabelEncoder()
# y = labelEncoder_y.fit_transform(y)
#==============================================================================

# split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)