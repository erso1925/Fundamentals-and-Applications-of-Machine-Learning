"""
Author:
Erşen Pamuk - 20160601039
CE 475 - LAB3
"""

import csv
import math
import matplotlib.pyplot as plt
import numpy as np

with open("team_big.csv") as f:
    csv_list = list(csv.reader(f))

age_list = np.array([])
exp_list = np.array([])
pow_list = np.array([])
sal_list = np.array([])

for row in csv_list:
    if row != csv_list[0]:
        age_list = np.append(age_list, int(row[4]))
        exp_list = np.append(exp_list, int(row[6]))
        pow_list = np.append(pow_list, float(row[7]))
        sal_list = np.append(sal_list, int(row[8]))


def instruction(a):
    first = a[0]
    last = a[len(a) - 1]

    if len(a) == 1:
        return first

    if first == last and len(a) == 2:
        return first

    if last > first:
        return first

    pivot = math.floor(len(a) / 2)

    if a[pivot - 1] > a[pivot]:
        return a[pivot]
    elif first > a[pivot]:
        return instruction(a[0:pivot])
    else:
        return instruction(a[pivot + 1:len(a)])


def r_square(y, y_predicted):
    RSS = 0
    TSS = 0

    for i in range(len(y)):
        RSS += (y[i] - y_predicted[i]) * (y[i] - y_predicted[i])
        TSS += (y[i] - np.mean(y)) * (y[i] - np.mean(y))

    r_squared = 1 - RSS / TSS

    print("R^2 score: " + str(r_squared))


ones = np.ones((1, len(age_list)))

X = np.column_stack((age_list, exp_list, pow_list))
y = sal_list

coeffs = np.linalg.inv(np.dot(X.T, X))
coeffs = np.dot(coeffs, X.T)
coeffs = np.dot(coeffs, y)

y_hat = np.dot(X, coeffs)

print("Showing original results:")
r_square(y, y_hat)

random_columns = np.random.randint(-1000, 1000, len(age_list))

X = np.column_stack((X, random_columns))

coeffs = np.linalg.inv(np.dot(X.T, X))
coeffs = np.dot(coeffs, X.T)
coeffs = np.dot(coeffs, y)

y_hat = np.dot(X, coeffs)

print("Showing results with an added random column:")
r_square(y, y_hat)

plt.title("Residual Error Plot")
plt.scatter(y_hat, np.abs(y - y_hat))
plt.show()
