import buy_sell_policy
import numpy as np
import optimizers

costPerTransaction = 0.0

def simulateNoiseSegmentationWithOptimizationLoop(data, optimizationIntervallLength):
    currentStocks  = 0
    initialCapital = max(data)   #just a reference value for now
    cash           = initialCapital
    dataNorm       = data/data[0]
    #start at N_longTerm since algo won't run before
    appendedActions    = np.zeros((optimizationIntervallLength))
    #appendedDeltaNorms = np.zeros((N_longTerm-1))
    portfolioValue     = np.ones((optimizationIntervallLength)) * (initialCapital)
    dataSinceLastBuy   =[]
    lastActionIt       = 0

    for it in range(optimizationIntervallLength, data.shape[0], 1):
        #get datas
        sigma, N_longTerm = optimizers.evaluateSigmaLongTermNumbers(data[it - optimizationIntervallLength : it])

        longTermData = dataNorm[it-(N_longTerm-1):it+1]
        if currentStocks == 1:
            dataSinceLastBuy = dataNorm[lastActionIt:it+1]
        else:
            dataSinceLastBuy = np.array([])

        #action             = buy_sell_policy.noiseDetectionAndSegmentationApproach(dataSinceLastBuy, longTermData, currentStocks, N_shortTerm, sigma)
        #action             = buy_sell_policy.noiseDetectionSingleValueApproach(dataSinceLastBuy, longTermData, currentStocks, sigma)
        action             = buy_sell_policy.dynamicThreshold(dataSinceLastBuy, longTermData, currentStocks, sigma, 1)
        appendedActions    = np.append(appendedActions,action)
        #take actions
        if (action == 1): #BUY
            currentStocks = 1
            lastActionIt  = it
            cash          = cash - (data[it]+costPerTransaction)
        if (action == 2): #SELL
            currentStocks = 0
            lastActionIt  = -1
            cash          = cash + (data[it]-costPerTransaction)
        portfolioValue = np.append(portfolioValue, cash+currentStocks*data[it])
    gain = cash + (data[it]*currentStocks) - initialCapital
    return gain, appendedActions, portfolioValue

def simulateNoiseSegmentationAlgo(data, sigma, N_longTerm, N_shortTerm, SELL_THRESHOLD):
    currentStocks  = 0
    initialCapital = max(data)   #just a reference value for now
    cash           = initialCapital
    dataNorm       = data/data[0]
    #start at N_longTerm since algo won't run before
    appendedActions    = np.zeros((N_longTerm-1))
    #appendedDeltaNorms = np.zeros((N_longTerm-1))
    portfolioValue     = np.ones((N_longTerm-1)) * (initialCapital)
    dataSinceLastBuy   =[]
    lastActionIt       = 0
    for it in range(N_longTerm-1, data.shape[0], 1):
        #get datas
        longTermData = dataNorm[it-(N_longTerm-1):it+1]
        if currentStocks == 1:
            dataSinceLastBuy = dataNorm[lastActionIt:it+1]
        else:
            dataSinceLastBuy = np.array([])

        #action             = buy_sell_policy.noiseDetectionAndSegmentationApproach(dataSinceLastBuy, longTermData, currentStocks, N_shortTerm, sigma)
        #action             = buy_sell_policy.noiseDetectionSingleValueApproach(dataSinceLastBuy, longTermData, currentStocks, sigma)
        action             = buy_sell_policy.dynamicThreshold(dataSinceLastBuy, longTermData, currentStocks, sigma, SELL_THRESHOLD)
        appendedActions    = np.append(appendedActions,action)
        #take actions
        if (action == 1): #BUY
            currentStocks = 1
            lastActionIt  = it
            cash          = cash - (data[it]+costPerTransaction)
        if (action == 2): #SELL
            currentStocks = 0
            lastActionIt  = -1
            cash          = cash + (data[it]-costPerTransaction)
        portfolioValue = np.append(portfolioValue, cash+currentStocks*data[it])
    gain = cash + (data[it]*currentStocks) - initialCapital
    return gain, appendedActions, portfolioValue

def simulateAlgorithmOnTestData(data, BUY_THRESHOLD, SELL_THRESHOLD, sigma, previousDataPointsUsed):
    currentStocks  = 0
    initialCapital = max(data)   #just a reference value for now
    capital        = initialCapital
    appendedActions    = np.zeros((previousDataPointsUsed+1))
    appendedDeltaNorms = np.zeros((previousDataPointsUsed+1))
    currentCapital = np.ones((previousDataPointsUsed+1)) * (initialCapital)
    for It in range(data.shape[0]-1-previousDataPointsUsed):
        timeIt = It + previousDataPointsUsed
        currentTimeStepIndex   = timeIt + 1
        action, deltaValueNorm = buy_sell_policy.decideOnActionComplex(data[(timeIt-previousDataPointsUsed):timeIt], data[currentTimeStepIndex], data[0], currentStocks, appendedActions, sigma, BUY_THRESHOLD, SELL_THRESHOLD)
        #action, deltaValueNorm = decideOnAction(data[timeIt-1], data[currentTimeStepIndex], currentStocks, BUY_THRESHOLD, SELL_THRESHOLD)
        appendedActions    = np.append(appendedActions,action)
        appendedDeltaNorms = np.append(appendedDeltaNorms,deltaValueNorm)
        if (action == 1): #BUY
            currentStocks = 1
            capital = capital - (data[currentTimeStepIndex]+costPerTransaction)
        if (action == 2): #SELL
            currentStocks = 0
            capital = capital + (data[currentTimeStepIndex]-costPerTransaction)
        currentCapital = np.append(currentCapital, capital+currentStocks*data[currentTimeStepIndex])
    return capital + (data[currentTimeStepIndex]*currentStocks) - initialCapital, appendedActions, appendedDeltaNorms, currentCapital
