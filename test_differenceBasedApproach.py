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

data = beyond_meat['<CLOSE>']#[100000:120000]
data = data.to_numpy()

plotData = False
optimization = False
kpis = True
KPI_optimization = True

longTermFrames = 51
shotTermFrames = 4
sellPercentage = 1.3

if plotData:
    tic = time.time()
    moneyMade, appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(data, longTermFrames, shotTermFrames, sellPercentage)  #10 long term, 3 short term, 10% sell threshold
    toc = time.time()
    print("simulation time", toc - tic)
    print("money Made", moneyMade, "     efficiency ", efficiency)
    print("percentage Money made", madeMoneyPercentage)

    #plot results
    plot_graphs.plotAlgoResults(data, appendActions, portfolioValue)

if kpis:
    dataKPI = beyond_meat['<CLOSE>']
    dataKPI = dataKPI.to_numpy()
    testCase_slowDecline      = dataKPI[1000:2600]
    testCase_bigRiseBigDrop   = dataKPI[7000:8000]
    testCase_steadyDecrease   = dataKPI[37000:44000]
    testCase_increaseWithDrop = dataKPI[18000:20500]
    testCase_noisyIncrease    = dataKPI[100000:120000]

    moneyMadeSlowDecline,    appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(testCase_slowDecline,      longTermFrames, shotTermFrames, sellPercentage)
    plot_graphs.plotAlgoResults(testCase_slowDecline, appendActions, portfolioValue)
    moneyMadeBigRiseBigDrop, appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(testCase_bigRiseBigDrop,   longTermFrames, shotTermFrames, sellPercentage)
    plot_graphs.plotAlgoResults(testCase_bigRiseBigDrop, appendActions, portfolioValue)
    moneyMadeSteadyDecrease, appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(testCase_steadyDecrease,   longTermFrames, shotTermFrames, sellPercentage)
    plot_graphs.plotAlgoResults(testCase_steadyDecrease, appendActions, portfolioValue)
    moneyMadeIncreaseWDrop,  appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(testCase_increaseWithDrop, longTermFrames, shotTermFrames, sellPercentage)
    plot_graphs.plotAlgoResults(testCase_increaseWithDrop, appendActions, portfolioValue)
    moneyMadeNoisyInrease,   appendActions, portfolioValue, madeMoneyPercentage, efficiency = differenceBasedApproach.simulateAlgo(testCase_noisyIncrease,    longTermFrames, shotTermFrames, sellPercentage)
    plot_graphs.plotAlgoResults(testCase_noisyIncrease, appendActions, portfolioValue)

    print("moneyMadeSlowDecline   ", moneyMadeSlowDecline, " vs Data Delta ", testCase_slowDecline[len(testCase_slowDecline)-1] - testCase_slowDecline[0])
    print("moneyMadeBigRiseBigDrop", moneyMadeBigRiseBigDrop, " vs Data Delta ", testCase_bigRiseBigDrop[len(testCase_bigRiseBigDrop)-1] - testCase_bigRiseBigDrop[0])
    print("moneyMadeSteadyDecrease", moneyMadeSteadyDecrease, " vs Data Delta ", testCase_steadyDecrease[len(testCase_steadyDecrease)-1] - testCase_steadyDecrease[0])
    print("moneyMadeIncreaseWDrop ", moneyMadeIncreaseWDrop, " vs Data Delta ", testCase_increaseWithDrop[len(testCase_increaseWithDrop)-1] - testCase_increaseWithDrop[0])
    print("moneyMadeNoisyInrease  ", moneyMadeNoisyInrease, " vs Data Delta ", testCase_noisyIncrease[len(testCase_noisyIncrease)-1] - testCase_noisyIncrease[0])

    if KPI_optimization:
        N_iterations = 5
        differenceBasedApproach.optimizeForMoneyMade(testCase_slowDecline, N_iterations)
        differenceBasedApproach.optimizeForMoneyMade(testCase_bigRiseBigDrop, N_iterations)
        differenceBasedApproach.optimizeForMoneyMade(testCase_steadyDecrease, N_iterations)
        differenceBasedApproach.optimizeForMoneyMade(testCase_increaseWithDrop, N_iterations)
        differenceBasedApproach.optimizeForMoneyMade(testCase_noisyIncrease, N_iterations)

if optimization:
    #optimize for parameters
    tic = time.time()
    differenceBasedApproach.optimizeForPercentage(data, 5)
    toc = time.time()
    print(toc-tic)
