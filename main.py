"""
Author: Erşen Pamuk
CE475 - Term Project
Honor Code:
On my honor, as an Izmir University of Economics student, I affirm that I will not give or receive any unauthorized
help on this project, and that all work will be my own. The effort in the project belongs completely to me.
"""

# Importing the libraries
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as skt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('ProjectData.csv')
first_column = dataset.x1.values
second_column = dataset.x2.values
third_column = dataset.x3.values
fourth_column = dataset.x4.values
fifth_column = dataset.x5.values
sixth_column = dataset.x6.values
seventh_column = dataset.y.values

x1_list = np.array(first_column)
x2_list = np.array(second_column)
x3_list = np.array(third_column)
x4_list = np.array(fourth_column)
x5_list = np.array(fifth_column)
x6_list = np.array(sixth_column)
y_list = np.array(seventh_column)

y_pred = np.array([])
y_pred1 = np.array([])
y_pred2 = np.array([])
y_pred3 = np.array([])
y_pred4 = np.array([])
y_pred5 = np.array([])
y_pred6 = np.array([])
ones_list = np.ones((len(x1_list)))

X = np.column_stack((ones_list, x1_list, x2_list, x3_list, x4_list, x5_list, x6_list))
y = y_list[0:100]

print("------------------------------------------------------------------------------")
print("The selected model Random Forest Regression is performing...\n")
time.sleep(2)
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X[0:100], y)
print("-------------------------------Predicted Values-------------------------------")
y_predicted = regressor.predict(X[100:120])
print(y_predicted)
print("------------------------------------------------------------------------------")
"""# Visualising the results
plt.scatter(y[80:100], y_predicted, color='red')
plt.title('Random Forest Regression Final Predictions')
plt.show()"""

# Splitting the dataset into the Training set and Test set
X = X[0:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# MSE Calculation function
def mse_calculation(y_pred, y_test):
    mse = 0
    for i in range(len(y_test)):
        mse += np.square(y_pred[i] - y_test[i])
    mse = mse / len(y_test)
    return mse


# R^2 Score Calculation function
def rsquare_score(y_train, y_pred):
    mean = np.mean(y_train)
    tss = np.sum((y_train - mean) ** 2)
    rss = np.sum((y_train - y_pred) ** 2)
    rsqr = (1 - (rss / tss))
    return rsqr


def linear_regression(X_train, y_train, X_test, y_test):
    # Training the Linear Regression model on the Training set
    print("------------------------------------------------------------------------------")
    print("Linear Regression is performing...\n")
    time.sleep(2)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    print("-------------------------------Predicted Test Values-------------------------------")
    y_pred = regressor.predict(X_train)
    y_pred1 = regressor.predict(X_test)
    print(y_pred1)
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the R2 score
    rsqr = rsquare_score(y_train, y_pred)
    print("The R^2 score for Linear Regression: " + str(rsqr))
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the MSE score
    mse = mse_calculation(y_pred1, y_test)
    print("MSE for Linear Regression: " + str(mse))
    print("------------------------------------------------------------------------------")
    # Visualising the results
    plt.scatter(y_test, y_pred1, color='red')
    plt.title('Linear Regression Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return


def polynomial_regression(X_train, y_train, X_test, y_test):
    # Training the Polynomial Regression model on the whole dataset
    print("------------------------------------------------------------------------------")
    print("2nd degree Polynomial Regression is performing...\n")
    time.sleep(2)
    # Predicting the Test set results
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg_2 = LinearRegression()
    poly_reg_2.fit(X_poly, y_train)
    print("---------------------Predicted Test Values for 2nd degree polynomial regression---------------------")
    y_pred2_1 = poly_reg_2.predict(poly_reg.fit_transform(X_train))
    y_pred2_2 = poly_reg_2.predict(poly_reg.fit_transform(X_test))
    print(y_pred2_2)
    print("------------------------------------------------------------------------------")
    # Calculating the R2 score
    r2_2 = rsquare_score(y_train, y_pred2_1)
    print("The R^2 score for 2nd degree polynomial regression: {:.4f}".format(r2_2))
    print("------------------------------------------------------------------------------")
    # Calculating the MSE score
    mse = mse_calculation(y_test, y_pred2_2)
    print("MSE for 2nd degree polynomial regression: " + str(mse))
    print("------------------------------------------------------------------------------")
    print("3rd degree Polynomial Regression is performing...\n")
    time.sleep(2)
    poly_reg = PolynomialFeatures(degree=3)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg_2 = LinearRegression()
    poly_reg_2.fit(X_poly, y_train)
    print("---------------------Predicted Test Values for 3rd degree polynomial regression---------------------")
    y_pred2_1_1 = poly_reg_2.predict(poly_reg.fit_transform(X_train))
    y_pred2_1_2 = poly_reg_2.predict(poly_reg.fit_transform(X_test))
    print(y_pred2_1_2)
    print("------------------------------------------------------------------------------")
    r2_3 = rsquare_score(y_train, y_pred2_1_1)
    print("The R^2 score for 3rd degree polynomial regression: {:.4f}".format(r2_3))
    print("------------------------------------------------------------------------------")
    mse = mse_calculation(y_test, y_pred2_1_2)
    print("MSE for 3rd degree polynomial regression: " + str(mse))
    print("------------------------------------------------------------------------------")
    print("4th degree Polynomial Regression is performing...\n")
    time.sleep(2)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg_2 = LinearRegression()
    poly_reg_2.fit(X_poly, y_train)
    print("---------------------Predicted Values for 4th degree polynomial regression---------------------")
    y_pred2_1_3 = poly_reg_2.predict(poly_reg.fit_transform(X_train))
    y_pred2_1_4 = poly_reg_2.predict(poly_reg.fit_transform(X_test))
    print(y_pred2_1_4)
    print("------------------------------------------------------------------------------")
    r2_4 = rsquare_score(y_train, y_pred2_1_3)
    print("The R^2 score for 4th degree polynomial regression: {:.4f}".format(r2_4))
    print("------------------------------------------------------------------------------")
    mse = mse_calculation(y_test, y_pred2_1_4)
    print("MSE for 4th degree polynomial regression: " + str(mse))
    print("------------------------------------------------------------------------------")
    # Visualising the results
    plt.scatter(y_test, y_pred2_2, color='red')
    plt.scatter(y_test, y_pred2_1_2, color='blue')
    plt.scatter(y_test, y_pred2_1_4, color='green')
    plt.title('Polynomial Regression Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return


def decision_tree_regression(X_train, y_train, X_test, y_test):
    header_list = np.array([])
    with open("ProjectData.csv") as f:
        csv_list = list(csv.reader(f))
    for row in csv_list:
        if row == csv_list[0]:
            header_list = np.append(header_list, row[0])
            header_list = np.append(header_list, row[1])
            header_list = np.append(header_list, row[2])
            header_list = np.append(header_list, row[3])
            header_list = np.append(header_list, row[4])
            header_list = np.append(header_list, row[5])
            header_list = np.append(header_list, row[6])
    feature_list = header_list.tolist()
    # Training the Decision Tree Regression model on the Training set
    print("------------------------------------------------------------------------------")
    print("Decision Tree Regression is performing with max_depth 1...\n")
    time.sleep(2)
    regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    print("-------------------------------Predicted Values-------------------------------")
    y_pred0 = regressor.predict(X_train)
    y_pred1 = regressor.predict(X_test)
    print(y_pred1)
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the R2 score
    rsqr = rsquare_score(y_train, y_pred0)
    print("The R^2 score for Decision Tree Regression is performing with max_depth 1: " + str(rsqr))
    print("------------------------------------------------------------------------------")
    # Calculating the MSE score
    mse = mse_calculation(y_pred1, y_test)
    print("MSE for Decision Tree Regression is performing with max_depth 1: " + str(mse))
    print("------------------------------------------------------------------------------")
    print("The feature importances: ", regressor.feature_importances_)
    r = skt.export_text(regressor, feature_names=feature_list)
    print(r)
    print("------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------")
    print("Decision Tree Regression is performing with max_depth 3...\n")
    time.sleep(2)
    regressor2 = DecisionTreeRegressor(random_state=0, max_depth=3)
    regressor2.fit(X_train, y_train)
    # Predicting the Test set results
    print("-------------------------------Predicted Values-------------------------------")
    y_pred2 = regressor2.predict(X_train)
    y_pred3 = regressor2.predict(X_test)
    print(y_pred3)
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    rsqr2 = rsquare_score(y_train, y_pred2)
    print("The R^2 score for Decision Tree Regression is performing with max_depth 3: " + str(rsqr2))
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    mse2 = mse_calculation(y_pred3, y_test)
    print("MSE for Decision Tree Regression is performing with max_depth 3: " + str(mse2))
    print("------------------------------------------------------------------------------")
    print("The feature importances: ", regressor2.feature_importances_)
    r2 = skt.export_text(regressor2, feature_names=feature_list)
    print(r2)
    print("------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------")
    print("Decision Tree Regression is performing with max_depth 7...\n")
    time.sleep(2)
    regressor3 = DecisionTreeRegressor(random_state=0, max_depth=7)
    regressor3.fit(X_train, y_train)
    # Predicting the Test set results
    print("-------------------------------Predicted Values-------------------------------")
    y_pred4 = regressor3.predict(X_train)
    y_pred5 = regressor3.predict(X_test)
    print(y_pred5)
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    rsqr3 = rsquare_score(y_train, y_pred4)
    print("The R^2 score for Decision Tree Regression is performing with max_depth 7: " + str(rsqr3))
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    mse3 = mse_calculation(y_pred5, y_test)
    print("MSE for Decision Tree Regression is performing with max_depth 7: " + str(mse3))
    print("------------------------------------------------------------------------------")
    print("The feature importances: ", regressor3.feature_importances_)
    r3 = skt.export_text(regressor3, feature_names=feature_list)
    print(r3)
    print("------------------------------------------------------------------------------")
    # Visualizing the results
    plt.scatter(y_test, y_pred1, color='green')
    plt.scatter(y_test, y_pred3, color='red')
    plt.scatter(y_test, y_pred5, color='blue')
    xpoints = plt.xlim()
    ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='-', color='gray')
    plt.title('Decision Tree: Predictions')
    plt.show()
    return


def support_vector_regression(X_train, y_train, X_test, y_test):
    print("------------------------------------------------------------------------------")
    print("Support Vector Regression is performing...\n")
    # Training the SVR model on the whole dataset
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)
    print("-------------------------------Predicted Values-------------------------------")
    # Predicting a new result
    y_pred0 = regressor.predict(X_train)
    y_pred1 = regressor.predict(X_test)
    print(y_pred1)
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the R2 score
    rsqr = rsquare_score(y_train, y_pred0)
    print("The R^2 score for Support Vector Regression: " + str(rsqr))
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the MSE score
    mse = mse_calculation(y_pred1, y_test)
    print("MSE for Support Vector Regression: " + str(mse))
    print("------------------------------------------------------------------------------")
    # Visualising the SVR results
    plt.scatter(y_test, y_pred1, color='red')
    plt.title('Support Vector Regression Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return


def random_forest_regression(X_train, y_train, X_test, y_test):
    # Training the Random Forest Regression model on the whole dataset
    print("------------------------------------------------------------------------------")
    print("Random Forest Regression is performing...\n")
    time.sleep(2)
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    print("-------------------------------Predicted Values-------------------------------")
    y_pred0 = regressor.predict(X_train)
    y_pred1 = regressor.predict(X_test)
    print(y_pred1)
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the R2 score
    rsqr = rsquare_score(y_train, y_pred0)
    print("The R^2 score for Random Forest Regression: " + str(rsqr))
    print("------------------------------------------------------------------------------")
    time.sleep(1)
    # Calculating the MSE score
    mse = mse_calculation(y_pred1, y_test)
    print("MSE for Random Forest Regression: " + str(mse))
    print("------------------------------------------------------------------------------")
    # Visualising the results
    plt.scatter(y_test, y_pred1, color='red')
    plt.title('Random Forest Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return

# linear_regression(X_train, y_train, X_test, y_test)
# polynomial_regression(X_train, y_train, X_test, y_test)
# decision_tree_regression(X_train, y_train, X_test, y_test)
# support_vector_regression(X_train, y_train, X_test, y_test)
# random_forest_regression(X_train, y_train, X_test, y_test)
