import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import simulate_algo
import plot_graphs
import optimizers
import helpers

tesla = pd.read_csv('Tesla_2020_09_15_2020_09_22.csv')
#siemens = pd.read_csv('../SIE-TDG_190101_201011.csv')
beyond_meat = pd.read_csv('data/US1.BYND_190513_201011.csv')
#tesla = pd.read_csv('Tesla_2020_09_08_2020_09_15.csv')
#tesla = pd.read_csv('Tesla_2020_09_01_2020_09_07.csv')
data  = tesla['Close']#[1200:1500]
data  = data.to_numpy()

beyond_meat = beyond_meat['<CLOSE>'][38000:42800]
data = beyond_meat.to_numpy()


#moneyMade, appendActions, appendedDeltas, currentCapital = simulate_algo.simulateAlgorithmOnTestData(data, 0.98666667, 0.909, 4.6, 18)
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 2.37, 20, 2)
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 0.185, 19, 2)
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 3.48, 6, 2)

tic = time.time()
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 1.17, 23, 1, 0)
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 1.75, 8, 1)

#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 1, 5, 1, 0.0010)
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 1.2, 12, 1, 0.5010)
#moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(data, 4.2, 6, 1, 6.25)
moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationWithOptimizationLoop(data, 100)

toc = time.time()
print(toc - tic)
#helpers.findLastSegmentBegin(data[60:100])

#import pdb; pdb.set_trace()
# buy_indexes  = np.where(appendActions == 1)
# sell_indexes = np.where(appendActions == 2)
# print(buy_indexes, sell_indexes)
print(moneyMade)

#plot_graphs.plotPolynomialFit(data[30:40], helpers.quadraticFitToData(data[30:40]))

plot_graphs.plotAlgoResults(data, appendActions, portfolioValue)

#plot_graphs.plotPartOfAlgoResults(data, appendActions, portfolioValue, 250, 400)

#optimizers.evaluateBuySellThreshold(data, 2.6, 18)

tic = time.time()
#optimizers.evaluateSigmaLongTermNumbersIntervalls(data)
#optimizers.evaluateSigmaLongTermNumbers(data)
#optimizers.bruteForceSigmaLongTermNumbersSellThreshold(data)
toc = time.time()
print(toc-tic)

#optimizers.bruteForceCalculateBestValues(data)
