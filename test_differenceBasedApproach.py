import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import differenceBasedApproach
import plot_graphs
import optimizers
import helpers

tesla = pd.read_csv('Tesla_2020_09_15_2020_09_22.csv')
#siemens = pd.read_csv('../SIE-TDG_190101_201011.csv')
beyond_meat = pd.read_csv('data/US1.BYND_190513_201011.csv')
#tesla = pd.read_csv('Tesla_2020_09_08_2020_09_15.csv')
#tesla = pd.read_csv('Tesla_2020_09_01_2020_09_07.csv')
data  = tesla['Close']#[1200:1500]

data = beyond_meat['<CLOSE>'][1000:2028]
data = data.to_numpy()


tic = time.time()
moneyMade, appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(data, 12, 1, 7.5)  #10 long term, 3 short term, 10% sell threshold
toc = time.time()
print("simulation time", toc - tic)
print("money Made", moneyMade, "     efficiency ", efficiency)
print("percentage Money made", madeMoneyPercentage)

#plot results
plot_graphs.plotAlgoResults(data, appendActions, portfolioValue)

#optimize for parameters
tic = time.time()
differenceBasedApproach.optimizeForPercentage(data, 5)
toc = time.time()
print(toc-tic)
