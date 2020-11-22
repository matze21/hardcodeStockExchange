import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize as optimize

import helpers

def predictNextValuePiecewise(dataSubSet):
        timePoints   = np.linspace(0,len(dataSubSet)-1, len(dataSubSet))

        initialVariables = np.array([2, 1, 0, 1])
        minimum = optimize.fmin(leastSquaresFunctionLinPiecewise, initialVariables, (timePoints, dataSubSet))

        nextValue = piecewiseLinearFunction(minimum[0], minimum[1], minimum[2], minimum[3], len(dataSubSet))
        res       = leastSquaresFunctionLinPiecewise(minimum, timePoints, dataSubSet)
        variance  = (np.square(res)).sum()/len(dataSubSet)
        delta = nextValue - dataSubSet[len(dataSubSet)-1]
        stdDeviation = np.sqrt(variance)

        return nextValue, stdDeviation, delta

def leastSquaresFunctionLinPiecewise(variables, x, y):
    x0 = variables[0]
    m1 = variables[1]
    t1 = variables[2]
    m2 = variables[3]
    error = 0
    for i in range(0, len(x)):
        x_cur = x[i]
        error = error + np.square(y[i] - piecewiseLinearFunction(x0, m1, t1, m2, x_cur))

    return error

def piecewiseLinearFunction(x0, m1, t1, m2, x):
    if x < x0:
        y = m1 * x + t1
    else:
        y = m2 * (x - x0)
    return y

tesla = pd.read_csv('Tesla_2020_09_15_2020_09_22.csv')
data  = tesla['Close'][0:20]
data  = data.to_numpy()
#
tic = time.time()
nextValueIM, stdDevIM, deltaIM = helpers.predictNextValuePiecewise(data)
toc = time.time()
print('imported module time =', toc - tic)
print('values = ', nextValueIM, stdDevIM, deltaIM)

tic = time.time()
nextValue, stdDev, delta = predictNextValuePiecewise(data)
toc = time.time()
print('my function time = ', toc -tic)
print('values = ', nextValue, stdDev, delta)
