import numpy as np
import time

costPerTransaction              = 0.0
visualDebugging                 = False
printTimestepsWhereAlgoMessedUp = False

if visualDebugging:
    import matplotlib.pyplot as plt

def simulateAlgo(data, N_longTerm, N_shortTerm, SELL_THRESHOLD):
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
    N_gainedMoney      = 0
    N_transactions     = 0  # 1 buy 1 sell
    Buy_Price          = 0
    Sell_Price         = 0
    Buy_it             = 0
    for it in range(N_longTerm-1, data.shape[0], 1):
        #get datas
        longTermData = dataNorm[it-(N_longTerm-1):it+1]
        if currentStocks == 1:
            dataSinceLastBuy = dataNorm[lastActionIt:it+1]
        else:
            dataSinceLastBuy = np.array([])

        action             = dynamicThreshold(dataSinceLastBuy, longTermData, currentStocks, N_shortTerm, SELL_THRESHOLD)
        appendedActions    = np.append(appendedActions,action)

        #take actions
        #print(action)
        if (action == 1): #BUY
            currentStocks = 1
            lastActionIt  = it
            cash          = cash - (data[it]+costPerTransaction)
            Buy_Price     = cash+currentStocks*data[it]
            Buy_it        = it
        if (action == 2): #SELL
            currentStocks = 0
            lastActionIt  = -1
            cash          = cash + (data[it]-costPerTransaction)
            Sell_Price    = cash
            N_transactions +=1
            if Sell_Price > Buy_Price:
                N_gainedMoney += 1
            else:
                if printTimestepsWhereAlgoMessedUp:
                    print("Bad Behavior from ", Buy_it, " to ", it, " delta ", Sell_Price - Buy_Price)
        portfolioValue = np.append(portfolioValue, cash+currentStocks*data[it])

    if currentStocks == 1:
        Sell_Price = cash + (data[it]- costPerTransaction)
        N_transactions +=1
        if Sell_Price > Buy_Price:
            N_gainedMoney += 1
        else:
            if printTimestepsWhereAlgoMessedUp:
                print("Bad Behavior from ", Buy_it, " to ", it, " delta ", Sell_Price - Buy_Price)

    if N_transactions > 0:
        madeMoneyPercentage = N_gainedMoney / N_transactions * 100
    else:
        madeMoneyPercentage = 0
    finalValue = cash + (data[it]*currentStocks)
    gain       = finalValue - initialCapital
    if visualDebugging:
        print("stock difference vs. money made", data[it]-data[0], " vs.", finalValue-initialCapital)
    efficiency = (finalValue - initialCapital) / (data[it] - data[N_longTerm-1])
    return gain, appendedActions, portfolioValue, madeMoneyPercentage, efficiency


#lastLongTrendValues needs to be greater than 2
def dynamicThreshold(dataPointsSinceBuyNorm, lastLongTrendValuesNorm, currentStocks, N_trendDetection, SELL_THRESHOLD):
    # look at the last changes of the stock
    N_longTerm         = lastLongTrendValuesNorm.shape[0]
    accumulatedDeltas  = 0
    curDelta           = lastLongTrendValuesNorm[N_longTerm-1] - lastLongTrendValuesNorm[N_longTerm-2]
    deltaStartEnd      = (lastLongTrendValuesNorm[N_longTerm-1] - lastLongTrendValuesNorm[0])
    for i in range(N_longTerm-2):
        accumulatedDeltas = accumulatedDeltas + abs(lastLongTrendValuesNorm[i] - lastLongTrendValuesNorm[i+1])
    averageDelta = accumulatedDeltas / i

    #check if stock is changing a lot, but the average value is staying relatively constant - lot of noise!
    isNoisy = abs(deltaStartEnd) < averageDelta * 3

    #get delta of the last couple time points = detection if rise is happening
    isRising         = True
    isFalling        = True
    for j in range(N_trendDetection):
        timeStamp = N_longTerm - 1 - j
        delta     = lastLongTrendValuesNorm[timeStamp] - lastLongTrendValuesNorm[timeStamp-1]
        isRising  = isRising and delta > 0
        isFalling = isFalling and delta < 0

    # for falling only the last delta is important
    isFalling = isFalling or curDelta < 0

    isChangeSignificant = abs(lastLongTrendValuesNorm[N_longTerm - 1] - lastLongTrendValuesNorm[N_longTerm -1 - N_trendDetection]) > averageDelta

    #check if we made money since the buy, if no, just sell if fall is significant ~10%
    N_sinceBuy    = dataPointsSinceBuyNorm.shape[0]
    deltaSinceBuy = 1.0
    if N_sinceBuy > 0:
        #use the highest value to determine how much we've been loosing
        highestValueNorm      = max(dataPointsSinceBuyNorm)
        deltaSinceBuy         = dataPointsSinceBuyNorm[N_sinceBuy-1] - dataPointsSinceBuyNorm[0]
        deltaMaxValuePercent  = (dataPointsSinceBuyNorm[N_sinceBuy-1] - highestValueNorm) /  highestValueNorm * 100
        if (deltaMaxValuePercent < -SELL_THRESHOLD):# or (deltaStartEnd<0):
            deltaSinceBuy = 1.0

    should_buy = not isNoisy and isChangeSignificant and isRising #and deltaStartEnd > 0
    should_sell = isFalling and deltaSinceBuy > 0 #and isChangeSignificant

    #define action
    action = 0
    if currentStocks == 1 and should_sell:
        action = 2
    if currentStocks == 0 and should_buy:
        action = 1

    if visualDebugging:
            fig2, (ax1, ax2) = plt.subplots(2,1)
            dataPointsT = np.linspace(0, N_longTerm-1, N_longTerm)
            ax1.plot(dataPointsT, lastLongTrendValuesNorm)
            ax1.set_title("data")

            print("--------------")
            print("action ", action)
            print("should_sell =", should_sell, "should_buy=", should_buy)
            print("isNoisy", isNoisy)
            print("isChangeSignificant", isChangeSignificant)
            print("isRising=", isRising, "isFalling=", isFalling)
            print("averageDelta=", averageDelta)
            print("DeltaStartEnd=", deltaStartEnd)

            if N_sinceBuy > 0:
                timeSinceBuy = np.linspace(0, N_sinceBuy-1, N_sinceBuy)
                ax2.plot(timeSinceBuy, dataPointsSinceBuyNorm)
                print("deltaSinceBuy", deltaInPercent, "%")
            plt.show()
    return action

def optimizeForPercentage(data, N_length):
    N_shortTerm = 3
    shortTermFrames= np.linspace(1,    N_shortTerm, N_shortTerm, dtype = np.int64)
    longTermFrames = np.linspace(3,    40,N_length, dtype = np.int64)
    sell_threshold = np.linspace(0.00, 15, N_length)
    z_grid         = np.zeros((N_shortTerm,N_length,N_length))
    percentages    = np.zeros((N_shortTerm,N_length,N_length))
    for i in range(N_shortTerm):
        for j in range(N_length):
            for k in range(N_length):
                tic = time.time()
                z_grid[i][j][k], appendActions, currentCapital, percentages[i][j][k], efficiency =simulateAlgo(data, longTermFrames[j], shortTermFrames[i], sell_threshold[k])
                toc = time.time()
                print(toc-tic)
    print("results = ", z_grid)
    result = np.where(z_grid == np.max(z_grid))
    print("best result ", z_grid[result])
    print("best short Term ", shortTermFrames[result[0]])
    print("best long Term frame ", longTermFrames[result[1]])
    print("best sell threshold ", sell_threshold[result[2]])

    print("percentage results = ", percentages)
    result = np.where(percentages == np.max(percentages))
    print("percentage best result ", percentages[result])
    print("percentage best short Term ", shortTermFrames[result[0]])
    print("percentage best long Term frame ", longTermFrames[result[1]])
    print("percentage best sell threshold ", sell_threshold[result[2]])
    #plot_graphs.plot2ParameterSpectrum(sigma,longTermFrames,z_grid, "sigma", "long term frames used")
    #print("best result indixes = ", result[0], result[1], result[2])

def optimizeForMoneyMade(data, N_length):
    N_shortTerm = 10
    shortTermFrames= np.linspace(1,    N_shortTerm, N_shortTerm, dtype = np.int64)
    longTermFrames = np.linspace(N_shortTerm,    40,N_length, dtype = np.int64)
    sell_threshold = np.linspace(0.00, 15, N_length)
    z_grid         = np.zeros((N_shortTerm,N_length,N_length))
    percentages    = np.zeros((N_shortTerm,N_length,N_length))
    for i in range(N_shortTerm):
        for j in range(N_length):
            for k in range(N_length):
                z_grid[i][j][k], appendActions, currentCapital, percentages[i][j][k], efficiency =simulateAlgo(data, longTermFrames[j], shortTermFrames[i], sell_threshold[k])
                toc = time.time()
    #print("results = ", z_grid)
    result = np.where(z_grid == np.max(z_grid))
    print("best result ", z_grid[result])
    print("best short Term ", shortTermFrames[result[0]])
    print("best long Term frame ", longTermFrames[result[1]])
    print("best sell threshold ", sell_threshold[result[2]])

    #print("percentage results = ", percentages)
    # result = np.where(percentages == np.max(percentages))
    # print("percentage best result ", percentages[result])
    # print("percentage best short Term ", shortTermFrames[result[0]])
    # print("percentage best long Term frame ", longTermFrames[result[1]])
    # print("percentage best sell threshold ", sell_threshold[result[2]])
