import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import simulate_algo
import plot_graphs
import optimizers
import helpers

my_dict={
 'siemens' : pd.read_csv('data/SIE-TDG_190101_201011.csv')['<CLOSE>'].to_numpy(),
 #'beyond_meat' : pd.read_csv('data/US1.BYND_190513_201011.csv')['<CLOSE>'].to_numpy()
 # 'basf'       : pd.read_csv('data/BAS-TDG_190101_201011.csv')['<CLOSE>'].to_numpy()
 # 'infineon'   : pd.read_csv('data/IFX-TDG_190101_201011.csv')['<CLOSE>'].to_numpy()
 # 'msft'       : pd.read_csv('data/US1.MSFT_190101_201011.csv')['<CLOSE>'].to_numpy()
 # 'conti'      : pd.read_csv('data/CON-TDG_191001_201011.csv')['<CLOSE>'].to_numpy()
 # 'google'     : pd.read_csv('data/US1.GOOGL_190101_201011.csv')['<CLOSE>'].to_numpy()
}



data_names =['siemens', 'beyond_meat', 'basf', 'infineon', 'msft','conti','google']
tic = time.tic()
for stock in data_names:
    moneyMade, appendActions, portfolioValue = simulate_algo.simulateNoiseSegmentationAlgo(my_dict[stock], 0.36, 20, 2)
    toc = time.toc()
    buy_indexes  = np.where(appendActions == 1)
    sell_indexes = np.where(appendActions == 2)
    print(buy_indexes, sell_indexes)
    print(moneyMade)
    print(toc-tic)

    plot_graphs.plotAlgoResults(my_dict[stock], appendActions, portfolioValue)
