"""
Author:
Ersen Pamuk - 20160601039
CE 475 - LAB2
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

with open("team_big.csv") as f:
    csv_list = list(csv.reader(f))

exp_list = np.array([])
sal_list = np.array([])

for row in csv_list:
    if row != csv_list[0]:
        exp_list = np.append(exp_list, int(row[6]))
        sal_list = np.append(sal_list, float(row[8]))

exp_list_1 = exp_list[0:20]
sal_list_1 = sal_list[0:20]
exp_list_2 = exp_list[20::]
sal_list_2 = sal_list[20::]

def simlin_coef(x, y):
    xAvg = np.mean(x)
    yAvg = np.mean(y)

    b1_1 = (((x - xAvg) * (y - yAvg)).sum()) / ((x - xAvg) * (x - yAvg)).sum()
    b0_1 = yAvg - (b1_1 * xAvg)

    return b0_1, b1_1

def simlin_plot(x, y, b0_1, b1_1):
    y_hat = b1_1 * x + b0_1
    plt.figure()

    plt.scatter(x, y, color='blue')
    plt.plot(x, y_hat, color='red')
    plt.title('Simple Linear Regression: Experience vs Salary')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    return y_hat

def r_square(y, y_pred):

    rss = 0
    tss = 0

    for i in range(len(y)):
        rss += (y[i]-y_pred[i])*(y[i]-y_pred[i])
        tss += (y[i]-np.mean(y))*(y[i]-np.mean(y))

    r_squared = 1 - rss/tss

    print("R^2 score: " + str(r_squared))

b0_1, b1_1 = simlin_coef(exp_list_1, sal_list_1)
b0_2, b1_2 = simlin_coef(exp_list_2, sal_list_2)

sal_pred_1 = simlin_plot(exp_list_1, sal_list_1, b0_2, b1_2)
sal_pred_2 = simlin_plot(exp_list_2, sal_list_2, b0_1, b1_1)

r_square(sal_list_1, sal_pred_1)
r_square(sal_list_2, sal_pred_2)
plt.show()
