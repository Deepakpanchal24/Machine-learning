# Importing necessary libraries
import numpy as np  # For handling arrays and numerical operations
import matplotlib.pyplot as plt  # For plotting (not used in this code, but useful for visualization)
import pandas as pd  # For handling datasets and dataframes

# Loading the dataset from the specified path
dataset = pd.read_csv(r'D:\Deepak\Prakash\7 30 batch prakash 2024\day 1 ML/Data.csv')
dataset  # Display the dataset

# Splitting the dataset into features (X) and target (y)
# X contains all columns except the last one (independent variables)
X = dataset.iloc[:, :-1].values

# y contains only the last column (dependent variable)
y = dataset.iloc[:, 3].values

# Handling missing values with SimpleImputer
from sklearn.impute import SimpleImputer

# Creating an imputer object with default strategy ('mean' by default)
imputer = SimpleImputer()
#(strategy="median") by default


imputer = SimpleImputer()
# Other strategies (optional):
# imputer = SimpleImputer(strategy="median")  # Fills missing values with median
# imputer = SimpleImputer(strategy="most_frequent")  # Fills missing values with the most frequent value
# imputer = SimpleImputer(strategy="constant", fill_value=0)  # Fills with a constant value like 0

# Fitting the imputer on columns 1 and 2 (assuming these have missing values)
imputer = imputer.fit(X[:, 1:3])

# Replacing missing values with calculated values (mean/median/mode/constant)
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data (converting text labels into numbers)
from sklearn.preprocessing import LabelEncoder

# Creating a LabelEncoder object for encoding categorical features in X
labelencoder_X = LabelEncoder()

# Applying label encoding to the first column of X (e.g., Country or Category)
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Encoding the target variable (y) if it's categorical
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Splitting the data into 80% training and 20% testing sets with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True)

# Other options for test size (commented out):
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=True)  # 75-25 split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=True)  # 70-30 split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=True)  # 85-15 split

# Note: 'random_state=True' ensures that the split is reproducible every time you run the code
