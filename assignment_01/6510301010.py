import numpy as np
import matplotlib.pyplot as plt

# linear Regression Function (least-squares)
def LINEST(x, y):
    """
    Perform linear regression on x and y data.
    Returns slope (m) and intercept (c) of the line.
    """
    # calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # calculate Sxx, Syy, and Sxy
    Sxx = np.sum((x - x_mean)**2)
    Syy = np.sum((y - y_mean)**2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))

    # calculate slope and intercept
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    return slope, intercept

if __name__ == '__main__':
    # step 1: Load the data
    data = np.loadtxt('data_input.txt', delimiter=',', skiprows=1) # skip header
    x = data[:, 0]  # extract x values
    y = data[:, 1]  # extract y values
    print(f"x: {x}")
    print(f"y: {y}")
    # step 2: Perform regression
    m, c = LINEST(x, y)

    # print the results
    print(f"Slope (m):      {m:.2f}")
    print(f"Intercept (c): {c:.2f}")

    # visualization using pyplot library
    plt.scatter(x, y, color='blue', label='Data points')  # Plot the data
    plt.plot(x, m * x + c, color='red', label='Regression line')  # Plot the regression line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression')
    plt.show()
