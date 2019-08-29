import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

pro = ts.pro_api('f80f16bef1bad9b0ac42056aa6343f1c6b74b1ce6e820e872f9b266d')

def stock_macd(ts_code):
    df = pro.daily(ts_code=ts_code)
    close = [float(x) for x in df['close']]
    # 调用talib计算指数移动平均线的值
    df['EMA12'] = talib.EMA(np.array(close), timeperiod=6)
    df['EMA26'] = talib.EMA(np.array(close), timeperiod=12)
    # 调用talib计算MACD指标
    df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
                                                              fastperiod=6, slowperiod=12, signalperiod=9)
    return df['MACD'], df['MACD'].iloc[-1]


def stock_rsi(ts_code):
    df = pro.daily(ts_code=ts_code)
    close = [float(x) for x in df['close']]
    df['RSI6'] = talib.RSI(np.array(close), timeperiod=6)
    df['RSI12'] = talib.RSI(np.array(close), timeperiod=12)
    df['RSI24'] = talib.RSI(np.array(close), timeperiod=24)
    dict = {'rsi6': df['RSI6'].iloc[-1], 'rsi12': df['RSI12'].iloc[-1], 'rsi24': df['RSI24'].iloc[-1]}
    return df['RSI6'], df['RSI12'], df['RSI24'], dict


def stock_mom(ts_code):
    df = pro.daily(ts_code=ts_code)
    close = [float(x) for x in df['close']]
    df['MOM'] = talib.MOM(np.array(close), timeperiod=5)
    return df['MOM'], df['MOM'].iloc[-1]


def stock_kdj(ts_code):
    df = pro.daily(ts_code=ts_code)
    low_list = df['low'].rolling(9, min_periods=9).min()
    low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    high_list = df['high'].rolling(9, min_periods=9).max()
    high_list.fillna(value=df['high'].expanding().max(), inplace=True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    dict = {'K': df['K'].iloc[-1], 'D': df['D'].iloc[-1], 'J': df['J'].iloc[-1]}
    return df['K'], df['D'], df['J'], dict


def stock_roc(ts_code, N):
    df = pro.daily(ts_code=ts_code)
    BX = df['close'].iloc[-1-N]
    AX = df['close'].iloc[-1]-BX
    ROC = AX/BX
    return ROC


def stock_sharpe(ts_code, N, r):
    # 调用part 1 中实现的两个计算波动率与平均收益率的函数
    mean_return = stock_mean_return(ts_code, N)
    volatility = stock_volatility(ts_code, N)
    sharpe = (mean_return-r)/volatility
    return sharpe



# 调用示例
# print(stock_macd('000001.SZ'))
# print (stock_rsi('000001.SZ'))
# print (stock_mom('000001.SZ'))
# print (stock_kdj('000001.SZ'))
# print(stock_roc('000001.SZ', 12))