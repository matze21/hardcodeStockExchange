import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np


def plotAlgoResults(data, appendActions, currentCapital):
    fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
    dataNorm  = data/data[0]
    ax1.plot(np.linspace(0, data.shape[0],data.shape[0]), dataNorm)
    ax1.set_title("normalized data")
    ax2.plot(np.linspace(0, data.shape[0],data.shape[0]), appendActions)
    ax2.set_title("actions")
    ax3.plot(np.linspace(0, data.shape[0],data.shape[0]), currentCapital)
    ax3.set_title("capital")
    plt.show()

def plotPartOfAlgoResults(data, appendActions, currentCapital, start, end):
    dataNorm  = data/data[0]
    printData = pd.DataFrame(dataNorm[start:end])
    printData['appendActions'] = np.transpose(appendActions[start:end])
    printData['delta'] = dataNorm[start:end] - dataNorm[start-1:end-1]
    printData['addedMoney'] = currentCapital[start:end] - currentCapital[start-1:end-1]
    print(printData)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot(np.linspace(start, end,end-start), dataNorm[start:end])
    ax1.set_title("normalized data")
    ax2.plot(np.linspace(start, end,end-start), appendActions[start:end])
    ax2.set_title("actions")
    ax3.plot(np.linspace(start, end,end-start), currentCapital[start:end]/currentCapital[0])
    ax3.set_title("normalized Capital")
    plt.show()

def plot2ParameterSpectrum(x,y,z, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_grid, y_grid = np.meshgrid(x,y)
    #ax.set_xlabel('sigma trusted')
    ax.set_xlabel(xlabel)
    #ax.set_ylabel('frames')
    ax.set_ylabel(ylabel)
    line=ax.plot_surface(x_grid,y_grid, z)
    plt.show()

def plotPointsAndSegments(seg1Data, seg2Data, m1, m2, t1, t2):
    data = np.append(seg1Data,seg2Data)
    timePointsSeg1 = np.linspace(0,len(seg1Data)-1, len(seg1Data))
    timePointsSeg2 = np.linspace(0,len(seg2Data)-1, len(seg2Data))
    #import pdb; pdb.set_trace()
    time = np.append(timePointsSeg1, timePointsSeg2 + len(timePointsSeg1))
    segments = m1*timePointsSeg1 + t1
    segments = np.append(segments, m2*timePointsSeg2 + t2)

    plt.plot(time, data, 'g^', time, segments,'r')
    plt.show()

def plotPolynomialFit(subData, coefficients):
    timePoints = np.linspace(0,len(subData)-1, len(subData))
    deg = len(coefficients) - 1
    fit = np.zeros(len(subData))
    for i in range(len(coefficients)):
        fit = fit + coefficients[i] * timePoints **(deg - i)

    plt.plot(timePoints, subData, 'g^', timePoints, fit,'r')
    plt.show()

def plot3TimeDependentVariables(data1, data2, data3, groundTruth, nameData1, nameData2, nameData3):
    timePoints1 = np.linspace(0,len(data1)-1, len(data1))
    timePoints2 = np.linspace(0,len(data2)-1, len(data2))
    timePoints3 = np.linspace(0,len(data3)-1, len(data3))
    timePoints = np.linspace(0,len(groundTruth)-1, len(groundTruth))
    ax1 = plt.subplot(4,1,1)
    ax1.plot(timePoints1, data1)
    ax1.set_title(nameData1)
    ax2 = plt.subplot(4,1,2)
    ax2.plot(timePoints2, data2)
    ax2.set_title(nameData2)
    ax3 = plt.subplot(4,1,3)
    ax3.plot(timePoints3, data3)
    ax3.set_title(nameData3)
    ax4 = plt.subplot(4,1,4)
    ax4.plot(timePoints, groundTruth)
    ax4.set_title('data')
    plt.show()
