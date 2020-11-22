import numpy as np
import plot_graphs
from scipy import optimize
import pwlf

def linearApproximateLastValues(lastDataValuesNorm):
    # values = m * (timestep) + t     get m and t
    timesteps       = np.linspace(0, lastDataValuesNorm.shape[0]-1, lastDataValuesNorm.shape[0])
    averageTimeStep = np.sum(timesteps) / timesteps.shape[0]
    averageValues   = np.sum(lastDataValuesNorm)/lastDataValuesNorm.shape[0]
    residualsTime   = timesteps - averageTimeStep
    redidualsValue  = lastDataValuesNorm - averageValues
    m = np.sum(np.dot(residualsTime, redidualsValue)) / np.matmul(residualsTime, np.transpose(residualsTime))
    t = averageValues - m * averageTimeStep

    #variance calculation (average sqaured error from approximation)
    variance = np.matmul((lastDataValuesNorm - (m * timesteps + t)), np.transpose(lastDataValuesNorm - (m * timesteps + t)))/lastDataValuesNorm.shape[0]
    stdDeviation = np.sqrt(variance)
    return m, t, stdDeviation, averageValues

def testFunction(x,y):
    return -(x-5)**2-(y-3)**2+5


def findLastSegmentBeginDiscont(dataSubSet):
    N_total = dataSubSet.shape[0]
    ErrorSegmentation = 1e10
    m_curSeg, m_lastSeg, t_curSeg, t_lastSeg, stdDev_curSeg, stdDev_lastSeg = 0, 0, 0, 0, 0, 0
    for i in range(N_total):
        segment1Data = dataSubSet[N_total - i:N_total]
        segment2Data = dataSubSet[0:N_total-i]
        #import pdb; pdb.set_trace()
        if segment1Data.shape[0]>1:
            m1, t1, stdDeviation1, averageValue1 = linearApproximateLastValues(segment1Data)
        else:
            stdDeviation1, m1,t1 = 0,0,0
        if segment2Data.shape[0]>1:
            m2, t2, stdDeviation2, averageValue2 = linearApproximateLastValues(segment2Data)
        else:
            stdDeviation2, m2,t2 = 0,0,0
        sumErrors = stdDeviation1**2 * segment1Data.shape[0] + stdDeviation2**2 * segment2Data.shape[0]
        if sumErrors < ErrorSegmentation:
            ErrorSegmentation = sumErrors
            m_curSeg  = m1
            m_lastSeg = m2
            t_curSeg  = t1
            t_lastSeg = t2
            stdDev_curSeg  = stdDeviation1
            stdDev_lastSeg = stdDeviation2
            curSegData  = segment1Data
            lastSegData = segment2Data

    #plot_graphs.plotPointsAndSegments(curSegData, lastSegData, m_curSeg, m_lastSeg, t_curSeg, t_lastSeg)
    return m_curSeg, m_lastSeg, t_curSeg, t_lastSeg, stdDev_curSeg, stdDev_lastSeg

#fit a piecewise function
def findLastSegmentBeginContinous(dataSubSet):
    N_total = dataSubSet.shape[0]
    ErrorSegmentation = 1e10
    m_curSeg, m_lastSeg, t_curSeg, t_lastSeg, stdDev_curSeg, stdDev_lastSeg = 0, 0, 0, 0, 0, 0
    for i in range(N_total):
        segment1Data = dataSubSet[N_total - i:N_total]
        segment2Data = dataSubSet[0:N_total-i]
        #import pdb; pdb.set_trace()

        if sumErrors < ErrorSegmentation:
            ErrorSegmentation = sumErrors
            m_curSeg  = m1
            m_lastSeg = m2
            t_curSeg  = t1
            t_lastSeg = t2
            stdDev_curSeg  = stdDeviation1
            stdDev_lastSeg = stdDeviation2
            curSegData  = segment1Data
            lastSegData = segment2Data
    #plot_graphs.plotPointsAndSegments(curSegData, lastSegData, m_curSeg, m_lastSeg, t_curSeg, t_lastSeg)

def quadraticFitToData(dataSubSet):
    timePoints   = np.linspace(0,len(dataSubSet)-1, len(dataSubSet))
    coefficients = np.polyfit(timePoints, dataSubSet,2)

    return coefficients

def predictNextValueQuadraticFit(dataSubSet):
    coefficients = quadraticFitToData(dataSubSet)
    deg          = len(coefficients) - 1
    nextValue    = 0
    lastValue    = 0
    for i in range(len(coefficients)):
        nextValue = nextValue + coefficients[i] * (len(dataSubSet)) **(deg - i)
        lastValue = lastValue + coefficients[i] * (len(dataSubSet)-1) **(deg - i)

    variance = 0
    timePoints   = np.linspace(0,len(dataSubSet)-1, len(dataSubSet))
    for i in range(len(coefficients)):
        variance = variance + np.matmul((coefficients[i] * timePoints **(deg - i)), np.transpose(coefficients[i] * timePoints **(deg - i)))
    variance = variance/len(dataSubSet)
    stdDeviation = np.sqrt(variance)

    delta = nextValue > lastValue

    return nextValue, stdDeviation, delta




def predictNextValuePiecewise(dataSubSet):
    timePoints   = np.linspace(0,len(dataSubSet)-1, len(dataSubSet))
    my_pwlf      = pwlf.PiecewiseLinFit(timePoints, dataSubSet)
    res          = my_pwlf.fit(2)
    variance     = (np.square(res)).sum()/len(dataSubSet)
    nextValue = my_pwlf.predict(len(dataSubSet))
    delta = nextValue - dataSubSet[len(dataSubSet)-1]
    stdDeviation = np.sqrt(variance)

    return nextValue, stdDeviation, delta

def predictPiecewise(dataSubSet):
    timePoints   = np.linspace(0,len(dataSubSet)-1, len(dataSubSet))
    my_pwlf      = pwlf.PiecewiseLinFit(timePoints, dataSubSet)
    res          = my_pwlf.fit(2)
    variance     = (np.square(res)).sum()/len(dataSubSet)
    nextValue    = my_pwlf.predict([len(dataSubSet), len(dataSubSet)+1])
    #import pdb; pdb.set_trace()
    delta        = nextValue[1] - nextValue[0]
    stdDeviation = np.sqrt(variance)
    values = my_pwlf.predict(timePoints)
    return nextValue, stdDeviation, delta, values
