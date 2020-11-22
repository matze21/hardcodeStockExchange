import simulate_algo
import helpers
import plot_graphs
import time
import numpy as np

def evaluateBuySellThreshold(data, sigmaTrusted, previousDataPointsUsed):
    length = 31
    x = np.linspace(0.0,3.7,length)
    y = np.linspace(0.0,3.7,length)
    z_grid = np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            z_grid[i][j], appendActions, appendedDeltas, currentCapital=simulate_algo.simulateAlgorithmOnTestData(data,x[i],y[i], sigmaTrusted, previousDataPointsUsed)

    plot_graphs.plot2ParameterSpectrum(x,y,z_grid, "buy threshold", "sell threshold")

def evaluateSigmaLongTermNumbers(data):
    length = 11
    sigma          = np.linspace(1.0,5.0,length)
    longTermFrames = np.linspace(3,23,length, dtype = np.int64)
    z_grid = np.zeros((length,length))
    tic = time.time()
    for i in range(length):
        for j in range(length):
            z_grid[i][j], appendActions, currentCapital=simulate_algo.simulateNoiseSegmentationAlgo(data, sigma[i], longTermFrames[j], 1, 6.25)

    # print("results = ", z_grid)
    result = np.where(z_grid == np.max(z_grid))
    toc = time.time()
    print(toc-tic)
    # print("best result ", z_grid[result])
    # print("best result indixes = ", result[0], result[1])
    # print("best sigma ", sigma[result[0]])
    # print("best long Term frame ", longTermFrames[result[1]])
    # plot_graphs.plot2ParameterSpectrum(sigma,longTermFrames,z_grid, "sigma", "long term frames used")
    return sigma[result[0]][0], longTermFrames[result[1]][0]

def evaluateSigmaLongTermNumbersIntervalls(data):
    interVallSize = 500
    LengthData = data.shape[0]
    print(LengthData)
    dataPointStart = interVallSize
    bestSigmas = []
    bestLengths = []
    results = []
    moneyMade = 0
    while dataPointStart < LengthData:
        tic = time.time()
        dataSubSet = data[dataPointStart-interVallSize:dataPointStart]
        length = 16
        sigma          = np.linspace(1.0,5.0,length)
        longTermFrames = np.linspace(3,23,length, dtype = np.int64)
        z_grid = np.zeros((length,length))
        for i in range(length):
            for j in range(length):
                z_grid[i][j], appendActions, currentCapital=simulate_algo.simulateNoiseSegmentationAlgo(dataSubSet, sigma[i], longTermFrames[j], 1, 6.25)
        result = np.where(z_grid == np.max(z_grid))
        moneyMade = moneyMade + z_grid[result][0]
        results.append(moneyMade)
        bestSigmas.append(sigma[result[0]][0])
        bestLengths.append(longTermFrames[result[1]][0])
        toc = time.time()
        print(toc-tic,dataPointStart, z_grid[result], sigma[result[0]], longTermFrames[result[1]])
        dataPointStart = dataPointStart + interVallSize
    plot_graphs.plot3TimeDependentVariables(bestSigmas, bestLengths, results, data, 'sigmas', 'lengths long term', 'money Made')


def bruteForceSigmaLongTermNumbersSellThreshold(data):
    length = 5
    sigma          = np.linspace(1.0,3.0,length)
    longTermFrames = np.linspace(3,23,length, dtype = np.int64)
    sell_threshold = np.linspace(0.00, 25, length)
    z_grid = np.zeros((length,length,length))
    for i in range(length):
        for j in range(length):
            for k in range(length):
                tic = time.time()
                z_grid[i][j][k], appendActions, currentCapital=simulate_algo.simulateNoiseSegmentationAlgo(data, sigma[i], longTermFrames[j], 1, sell_threshold[k])
                toc = time.time()
                print(toc-tic)
    print("results = ", z_grid)
    result = np.where(z_grid == np.max(z_grid))
    print("best result ", z_grid[result])
    print("best sigma ", sigma[result[0]])
    print("best long Term frame ", longTermFrames[result[1]])
    print("best sell threshold ", sell_threshold[result[2]])
    #plot_graphs.plot2ParameterSpectrum(sigma,longTermFrames,z_grid, "sigma", "long term frames used")
    #print("best result indixes = ", result[0], result[1], result[2])

def bruteForceCalculateBestValues(data):
    length = 21
    sigma          = np.linspace(0.0,3.7,length)
    longTermFrames = np.linspace(10,30,length, dtype = np.int64)
    shortTermFrames= np.linspace(2,10,8, dtype = np.int64)
    print(longTermFrames, shortTermFrames)
    z_grid = np.zeros((length,length, 8))
    for i in range(length):
        for j in range(length):
            for k in range(8):
                z_grid[i][j][k], appendActions, currentCapital=simulate_algo.simulateNoiseSegmentationAlgo(data,sigma[i], longTermFrames[j], shortTermFrames[k])
    print("results = ", z_grid)
    result = np.where(z_grid == np.max(z_grid))
    print("best result ", z_grid[result])
    print("best result indixes = ", result)
    print("best sigma ", sigma[result[0]])
    print("best long Term frame ", longTermFrames[result[1]])
    print("best short term frame ", shortTermFrames[result[2]])
