# Author:
# Erşen Pamuk
# 20160601039

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

# First Function
def simlin_coeff(x, y):

    # y = b1x + b0
    # Average of x is defined by xMean
    # Average of y is defined by yMean
    # b0 = yMean - b1 * xMean
    # b1 = Sum((xi-xMean)(yi-yMean)) / Sum((xi-xMean)^2)

    # Average calculations of x and y:
    xMean = x.mean()
    yMean = y.mean()

    # Calculation of b1 by numpy
    b1 = (((x-xMean)*(y-yMean)).sum())/((x-xMean)*((x-xMean))).sum()

    # Calculation of b0 by numpy
    b0 = yMean - (b1*xMean)

    return b0, b1

# Second Function
def simlin_plot(x, y, b0, b1):

    # Plotting the x and points
    plt.scatter(x, y, color='blue')

    # Plotting the constructed regression line
    plt.plot(x, (b1 * x + b0), color='red')

    # Setting the x-axis
    plt.xlabel('Experience')

    # Setting the y-axis label
    plt.ylabel('Salary')

    # Setting the title
    plt.title('Simple Linear Regression: Experience and Salary')

    plt.show()

    # Main function
if __name__ == "__main__":
    # Reading the data from team_big.csv
    myData = np.genfromtxt('team_big.csv', delimiter=',')

    # Extracting the Experience into x
    x = myData[1:, 6]

    # Extracting the Salary into y
    y = myData[1:, 8]

    # Calling the first function
    b0, b1 = simlin_coeff(x, y)

    # Calling to plot
    simlin_plot(x, y, b0, b1)
