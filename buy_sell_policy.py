import helpers
import numpy as np
import time
import matplotlib.pyplot as plt

visualDebugging = False

def dynamicThreshold(dataPointsSinceBuyNorm, lastLongTrendValuesNorm, currentStocks, sigma, SELL_THRESHOLD):
    #noise detection
    N_longTerm                         = lastLongTrendValuesNorm.shape[0]
    trendWithoutShortTermTrend         = lastLongTrendValuesNorm[0:N_longTerm-1]
    m_d, t_d, stdDev_d, av_d                  = helpers.linearApproximateLastValues(trendWithoutShortTermTrend)
    m, t, stdDeviation, averageValue_longTerm = helpers.linearApproximateLastValues(lastLongTrendValuesNorm)

    #predict current value with long term trend
    #import pdb; pdb.set_trace()
    deltaLastValues    = lastLongTrendValuesNorm[N_longTerm-1] - lastLongTrendValuesNorm[N_longTerm-2]
    isNewTrendDetected = abs(deltaLastValues) > sigma * stdDev_d    #criteria to check if trend changed significantlz
    isLongTermTrendPositive = lastLongTrendValuesNorm[N_longTerm-1] - lastLongTrendValuesNorm[0] > 0

    curDelta = 0
    for i in range(N_longTerm-2):
        curDelta = curDelta + abs(lastLongTrendValuesNorm[i] - lastLongTrendValuesNorm[i+1])
    averageDelta = curDelta / i
    isNoisy = deltaLastValues < averageDelta and (lastLongTrendValuesNorm[N_longTerm-1] - lastLongTrendValuesNorm[0]) < averageDelta


    should_buy = (deltaLastValues > 0 and isNewTrendDetected and isLongTermTrendPositive) and not isNoisy #or m > 0

    predictedValue    = m_d * N_longTerm + t_d
    deltaPredictToCur = lastLongTrendValuesNorm[N_longTerm-1] - predictedValue

    N_sinceBuy = dataPointsSinceBuyNorm.shape[0]
    deltaSinceBuy = 1.0
    if N_sinceBuy > 0:
        deltaSinceBuy = dataPointsSinceBuyNorm[N_sinceBuy-1] - dataPointsSinceBuyNorm[0]
        deltaInPercent = deltaSinceBuy /  dataPointsSinceBuyNorm[0] * 100
        isTrendNegative = False
        if N_sinceBuy > 1:
            isTrendNegative = isTrendNegative or (dataPointsSinceBuyNorm[N_sinceBuy-1] - dataPointsSinceBuyNorm[N_sinceBuy-2]) < 0
        if N_sinceBuy > 2:
            isTrendNegative = isTrendNegative or (dataPointsSinceBuyNorm[N_sinceBuy-2] - dataPointsSinceBuyNorm[N_sinceBuy-3]) < 0

        if (deltaInPercent < -averageDelta): #sell if price drops more than 5%
        #if (deltaInPercent < -30): #and isTrendNegative:
        #if isTrendNegative:
            deltaSinceBuy = 1.0


    should_sell = ((deltaPredictToCur + stdDev_d) < 0 or (deltaLastValues < stdDeviation)) and deltaSinceBuy > 0

    #define action
    action = 0
    if currentStocks == 1 and should_sell:
        action = 2
    if currentStocks == 0 and should_buy:
        action = 1

    if visualDebugging:
            fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
            dataPointsT = np.linspace(0, N_longTerm-1, N_longTerm)
            timePoints  = np.linspace(0, N_longTerm, N_longTerm+1)
            trendValues = m * timePoints + t
            trendWS = m_d * timePoints + t_d
            ax1.plot(timePoints, trendValues, dataPointsT, lastLongTrendValuesNorm)
            ax1.set_title("full fit")
            ax2.plot(timePoints, trendWS, dataPointsT, lastLongTrendValuesNorm)
            ax2.set_title("fit without last one")

            print("--------------")
            print("action ", action)
            print("m /////// stdDeviation", m, stdDeviation)
            print("fit without last value m + stdDev", m_d, stdDev_d)
            print("deltaPredToCur", deltaPredictToCur)
            print("deltaLastValues", deltaLastValues)
            print("isNewTrend, isNoisy", isNewTrendDetected, isNoisy)
            if N_sinceBuy > 0:
                timeSinceBuy = np.linspace(0, N_sinceBuy-1, N_sinceBuy)
                ax3.plot(timeSinceBuy, dataPointsSinceBuyNorm)
                print("deltaSinceBuy, isTrendNegative", deltaSinceBuy, isTrendNegative)
            plt.show()
    return action

# lastLongTrendValues       last values, length depends on how many dada points are needed for a long term trend evaluation
def noiseDetectionAndSegmentationApproach(dataPointsSinceBuyNorm, lastLongTrendValuesNorm, currentStocks, N_shortTerm, sigma):
    #noise detection
    N_longTerm                         = lastLongTrendValuesNorm.shape[0]
    longTermTrendWithoutShortTermTrend = lastLongTrendValuesNorm[0:N_longTerm-N_shortTerm]
    shortTermTrend                     = lastLongTrendValuesNorm[N_longTerm-N_shortTerm:N_longTerm]
    m_longTerm, t_longTerm, stdDeviation_longTerm, averageValue_longTerm     = helpers.linearApproximateLastValues(longTermTrendWithoutShortTermTrend)
#    m_shortTerm, t_shortTerm, stdDeviation_shortTerm, averageValue_shortTerm = helpers.linearApproximateLastValues(shortTermTrend)

    #predict current value with long term trend
    predLongTerm       = m_longTerm * N_longTerm + t_longTerm
    lastValueIsOutlier = abs(predLongTerm - lastLongTrendValuesNorm[N_longTerm-1]) > sigma * stdDeviation_longTerm    #criteria to check if trend changed significantlz
    #lastValueIsOutlier = abs(lastLongTrendValuesNorm[N_longTerm-2] - lastLongTrendValuesNorm[N_longTerm-1]) > sigma * stdDeviation_longTerm
    #lastDelta          = lastLongTrendValuesNorm[N_longTerm-1] - lastLongTrendValuesNorm[N_longTerm-2]
    lastDelta          = lastLongTrendValuesNorm[N_longTerm-1] - predLongTerm
    varianceBuy        = (lastValueIsOutlier and lastDelta > 0) #or m_longTerm > 0
    varianceSell       = (lastValueIsOutlier and lastDelta < 0) #or m_longTerm < 0

    if visualDebugging:
        fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
        timeSinceBuy = np.linspace(0,len(lastLongTrendValuesNorm)-1, len(lastLongTrendValuesNorm))
        ValuesSincebuy = m_longTerm * timeSinceBuy + t_longTerm
        ax1.plot(timeSinceBuy, ValuesSincebuy, timeSinceBuy, lastLongTrendValuesNorm)

    #look at values since buy
    N_sinceBuy   = dataPointsSinceBuyNorm.shape[0]
    lostSinceBuy = False
    if N_sinceBuy > 3:
        m_sinceBuy, t_sinceBuy, stdDeviation_sinceBuy, averageValue_sinceBuy     = helpers.linearApproximateLastValues(dataPointsSinceBuyNorm)
        deltaSinceBuy = dataPointsSinceBuyNorm[N_sinceBuy-1] - dataPointsSinceBuyNorm[0]
        lostSinceBuy  = dataPointsSinceBuyNorm[N_sinceBuy-1] - dataPointsSinceBuyNorm[0] < 0
        lostSinceBuy = (lostSinceBuy) or m_sinceBuy < 0
        timeSinceBuy = np.linspace(0,len(dataPointsSinceBuyNorm)-1, len(dataPointsSinceBuyNorm))
        ValuesSincebuy = m_sinceBuy * timeSinceBuy + t_sinceBuy
        if visualDebugging:
            ax2.plot(timeSinceBuy, ValuesSincebuy)
            print('stdDeviation_sinceBuy', stdDeviation_sinceBuy)

    #piecewise fit to determine short term trends
    isNewSegmentPositive = False
    isNewSegmentNegative = False
    if N_longTerm > 1:
        #predictedValue, std_quadFit, predDelta = helpers.predictNextValuePiecewise(lastLongTrendValuesNorm)
        predictedValue, std_quadFit, predDelta, values = helpers.predictPiecewise(lastLongTrendValuesNorm)
        if visualDebugging:
            ax3.plot(np.linspace(0,len(lastLongTrendValuesNorm)-1, len(lastLongTrendValuesNorm)), values,np.linspace(0,len(lastLongTrendValuesNorm)-1, len(lastLongTrendValuesNorm)), lastLongTrendValuesNorm)

    #realDelta = lastLongTrendValuesNorm[N_longTerm-1]-(predictedValue - predDelta)
    #isQuadTrendTrustworthy = abs(predictedValue - lastLongTrendValuesNorm[N_longTerm-1]) < sigma*std_quadFit
        isQuadTrendTrustworthy = std_quadFit < stdDeviation_longTerm
        isQuadTrendTrustworthy = True
        isNewSegmentPositive = predDelta > 0 and isQuadTrendTrustworthy
        isNewSegmentNegative = predDelta < 0 and isQuadTrendTrustworthy
        if visualDebugging:
            print('std_quadFit', std_quadFit)

    #buy strategy
    should_buy = varianceBuy and isNewSegmentPositive #and (m_shortTerm > m_longTerm)

    #sell strategy
    N_sinceBuy = dataPointsSinceBuyNorm.shape[0]
    madeMoney = False
    if N_sinceBuy>1:
        deltaInPercent = (dataPointsSinceBuyNorm[N_sinceBuy-1] - dataPointsSinceBuyNorm[0])/dataPointsSinceBuyNorm[0] * 100
        madeMoney = deltaInPercent > 0
        if (deltaInPercent < -5): #sell if price drops more than 5%
            madeMoney = True
    should_sell = ((varianceSell or  isNewSegmentNegative) or lostSinceBuy) and madeMoney #and (m_shortTerm < m_longTerm)

    #define action
    action = 0
    if currentStocks == 1 and should_sell:
        action = 2
    if currentStocks == 0 and should_buy:
        action = 1

    if visualDebugging:
        print('isOutlier',lastValueIsOutlier)
        print('stdDev_longTerm', stdDeviation_longTerm)
        print('isNewSegmentPos, isNewSegmentNeg', isNewSegmentPositive, isNewSegmentNegative)
        print('action', action)
        plt.show()

    return action

def noiseDetectionSingleValueApproach(dataPointsSinceBuyNorm, lastLongTrendValuesNorm, currentStocks, sigma):
    #noise detection
    N_longTerm                         = lastLongTrendValuesNorm.shape[0]
    N_shortTerm = 1
    longTermTrendWithoutShortTermTrend = lastLongTrendValuesNorm[0:N_longTerm-N_shortTerm]
    shortTermTrend                     = lastLongTrendValuesNorm[N_longTerm-N_shortTerm:N_longTerm]
    m_longTerm, t_longTerm, stdDeviation_longTerm, averageValue_longTerm     = helpers.linearApproximateLastValues(longTermTrendWithoutShortTermTrend)
    #m_shortTerm, t_shortTerm, stdDeviation_shortTerm, averageValue_shortTerm = helpers.linearApproximateLastValues(shortTermTrend)

    #predict current value with long term trend
    predLongTerm       = m_longTerm * N_longTerm + t_longTerm

    isNewTrendDetected = abs(predLongTerm - lastLongTrendValuesNorm[N_longTerm-1]) > sigma * stdDeviation_longTerm    #criteria to check if trend changed significantlz

    #buy strategy
    should_buy = isNewTrendDetected and (lastLongTrendValuesNorm[N_longTerm-1] > predLongTerm) and (lastLongTrendValuesNorm[N_longTerm-1] > lastLongTrendValuesNorm[N_longTerm-3]+ stdDeviation_longTerm)

    #sell strategy
    should_sell = isNewTrendDetected and (lastLongTrendValuesNorm[N_longTerm-1] < predLongTerm) and (lastLongTrendValuesNorm[N_longTerm-1] < lastLongTrendValuesNorm[N_longTerm-3] - stdDeviation_longTerm)

    #define action
    action = 0
    if currentStocks == 1 and should_sell:
        action = 2
    if currentStocks == 0 and should_buy:
        action = 1

    return action

# function to decide whether to buy or sell
def decideOnAction(previousDataPoint, currentDataPoint, normalizeValue, currentStocks, BUY_THRESHOLD, SELL_THRESHOLD):
    action         = 0
    deltaValue     = currentDataPoint - previousDataPoint
    deltaValueNorm = deltaValue / normalizeValue * 100 #in percent
    if((deltaValueNorm > BUY_THRESHOLD) and (currentStocks == 0)):
        action = 1
    if(deltaValueNorm < -SELL_THRESHOLD and currentStocks == 1):
        action = 2
    return action, deltaValueNorm

def decideOnActionComplex(lastDataValues, currentValue, normalizeValue, currentStocks, appendedActions, sigma, BUY_THRESHOLD, SELL_THRESHOLD):
    action = 0
    dataValuesNorm = lastDataValues.to_numpy() / normalizeValue
    currentValueNorm = currentValue / normalizeValue

    dataValuesNorm = np.append(dataValuesNorm,currentValueNorm)
    m, t, stdDeviation, averageValue = helpers.linearApproximateLastValues(dataValuesNorm)
    # is new value within trend of linear approximation
    predictedNewValue = m * (dataValuesNorm.shape[0]) + t   #time steps start at 0 and end at shape[0]-1
    #difference        = currentValueNorm - predictedNewValue

    lastThreeDataValuesNorm = dataValuesNorm[dataValuesNorm.shape[0]-3:dataValuesNorm.shape[0]]
    m_short, t_short, stdDeviation_short, averageValue_short = helpers.linearApproximateLastValues(lastThreeDataValuesNorm)

    threshold_percent = 0.1
    threshold_slope = threshold_percent/100
    #slope_inPercent = m * 100 #1 % is a lot
    isShortTrendinVar = abs(lastThreeDataValuesNorm[0]-currentValueNorm) > stdDeviation#5*stdDeviation_short
    trustShortTrend = stdDeviation_short < 0.1*stdDeviation and isShortTrendinVar
    trustShortTrend = isShortTrendinVar
    pos_shortTrend = m > threshold_slope and trustShortTrend
    neg_shortTrend = m < -threshold_slope and trustShortTrend
    #import pdb; pdb.set_trace()

    longTermNegativeTrend = (m<-threshold_slope) and (dataValuesNorm[0]>(currentValueNorm+sigma*stdDeviation))
    longTermPositiveTrend = (m>threshold_slope) and (dataValuesNorm[0]<(currentValueNorm-0.7*sigma*stdDeviation))

#     isNewValuePlausible     = abs(difference) < (3*stdDeviation)
#     isNewValueVeryPlausible = abs(difference) < stdDeviation
    isNewValueVeryPlausible = (currentValueNorm < (predictedNewValue+stdDeviation)) or (currentValueNorm > (predictedNewValue-stdDeviation))


#     #shorttermTrend
#     mShort, tShort, stdDeviationShort, averageValueShort = linearApproximateLastValues(dataValuesNorm[dataValuesNorm.shape[0]-5:dataValuesNorm.shape[0])])
#     predictedNewValueShort = mShort * (5) + tShort
#     shortTermNegativeTrend = m<0

#     if longTermNegativeTrend or longTermPositiveTrend:
#         if currentStocks == 0:
#             if longTermPositiveTrend and isNewValueVeryPlausible:# or (isNewValuePlausible and difference > 0)):
#                 action = 1
#         if currentStocks == 1:
# #             if longTermPositiveTrend and ((not isNewValuePlausible) and difference < 0):
# #                 action = 2
#             if longTermNegativeTrend and isNewValueVeryPlausible:
#                 action = 2

    if pos_shortTrend and currentStocks ==0:# and not longTermNegativeTrend:
        action = 1
    if neg_shortTrend and currentStocks ==1:# and not longTermPositiveTrend:
        action = 2
#    else:
#         if (not isNewValuePlausible):     #increase the threshold if the new Value is very far away from the last couple values
#    BUY_THRESHOLD = 2
#    SELL_THRESHOLD= 2
#     deltaValueNorm  = (currentValueNorm - dataValuesNorm[dataValuesNorm.shape[0]-2])*100 # in percent #currentValue = dataValues[shape-1]
#     if((deltaValueNorm > BUY_THRESHOLD) and (currentStocks == 0)):
#         action = 1
#     if(deltaValueNorm < -SELL_THRESHOLD and currentStocks == 1):
#         action = 2

    if appendedActions.shape[0]>4:
        isActionToggling = appendedActions[appendedActions.shape[0]-1] == 2
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-2] == 2
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-3] == 2
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-4] == 2
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-1] == 1
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-2] == 1
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-3] == 1
        isActionToggling = isActionToggling or appendedActions[appendedActions.shape[0]-4] == 1
        #if isActionToggling:
           # action = 0


    trend=0
    if longTermNegativeTrend:
        trend = -1
    if longTermPositiveTrend:
        trend = 1
    return action,trend
