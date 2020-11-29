import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

tickers = ['AAPL', 'ORCL', 'TSLA', 'BYND', 'EOAN.DE', 'EX GOOGLE', 'AMZN', 'CSCO','COST','CONTINENTAL AG', 'SIEMENA AG', 'BMW AG', 'ALLIANZ']


for t in tickers:
    start_date = date(2020, 9, 30)
    end_date   = date(2020, 10, 6)
    stockData = yf.download(t, start=start_date, end=end_date, interval = '1m')
    # for i in daterange(start_date + timedelta(1), end_date):
    #     day = yf.download(t, start=start_date, end=end_date, interval = '1m')
    #     stockData = stockData.append(day)#, ignore_index = True)
    stringStart = start_date.strftime("%m_%d_%Y")
    stringEnd   = end_date.strftime("%m_%d_%Y")
    stockData.to_csv(t + '_' + stringStart +'_' + stringEnd + '.csv')
    if tickers.index(t) == 0:
        normalizedData = stockData['Close']/stockData['Close'][0]
print(stockData)


#tsla_mod = tsla_test.drop(tsla_test.index[tsla_test.shape[0]-1])

#tsla_mod.to_csv('Tesla_2020_09_01_2020_09_07.csv')
