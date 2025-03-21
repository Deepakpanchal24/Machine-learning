# Import required libraries
import numpy as np  # For numerical operations (though not directly used here)
import matplotlib.pyplot as plt  # For data visualization (plotting graphs)
import pandas as pd  # For handling datasets (data manipulation)

# Import the dataset
dataset = pd.read_csv(r'D:\Deepak\Datascience\Code\4 march Simple linear regression\4th - slr\4th - slr\SIMPLE LINEAR REGRESSION\Salary_Data.csv')
# Loads the Salary_Data.csv file into a DataFrame called 'dataset'

# Split the data into the independent variable (Years of Experience) and dependent variable (Salary)
X = dataset.iloc[:, :-1].values  # Extracts all rows and all columns except the last one as features (Years of Experience)
y = dataset.iloc[:, 1].values  # Extracts the second column (Salary) as the target variable

# As the dependent variable is continuous, we use a regression algorithm.
# Since the dataset has two attributes, Simple Linear Regression (SLR) is suitable.

# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# random_state=0 ensures reproducibility (same split every time the code runs)

# Import the Linear Regression model from sklearn
from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression class
regressor = LinearRegression()

# Train (fit) the Simple Linear Regression model using the training data
regressor.fit(X_train, y_train)

#Linear regression = Algorithm
#regressor = model

# Predict the target values (salaries) for the test data
y_pred = regressor.predict(X_test)

# Visualize the training set results
plt.scatter(X_train, y_train, color='red')  # Plot the actual training data points in red
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Plot the regression line
plt.title('Salary vs Experience (Training set)')  # Graph title
plt.xlabel('Years of Experience')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()  # Display the plot

# Visualize the test set results
plt.scatter(X_test, y_test, color='red')  # Plot the actual test data points in red
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Plot the same regression line as above
plt.title('Salary vs Experience (Test set)')  # Graph title
plt.xlabel('Years of Experience')  # X-axis label
plt.ylabel('Salary')  # Y-axis label
plt.show()  # Display the plot

# Extract the slope (coefficient) of the linear regression line
m = regressor.coef_  # Slope (rate of change of Salary with respect to Experience)
m

# Extract the intercept of the linear regression line
c = regressor.intercept_  # Intercept (Salary when Experience = 0)
c

# Check for overfitting (low bias, high variance) by evaluating performance on the training set
bias = regressor.score(X_train, y_train)  # R² score (explained variance) for training data


# Check for underfitting (high bias, low variance) by evaluating performance on the test set
variance = regressor.score(X_test, y_test)  # R² score (explained variance) for test data
