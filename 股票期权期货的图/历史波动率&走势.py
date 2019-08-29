import pyecharts
import tushare as ts
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

ts.set_token('a93f250e15311901b51e097c305d0c14d1961dd5113fa09d430b2e6b')
pro = ts.pro_api()
'''
历史走势图（待获取数据后根据数据结构编写）
预计效果：K线图，横坐标trade_date，纵坐标price。
包含开盘价、收盘价、最高/低价、成交量，可分日、周、月、年查看某产品的走势（详细程度要看获取的数据）。
需要数据：open、close、high、low、volume、trade_date
'''  
#历史波动率
def volatility(seq):#seq是价格序列 需要数据：期权/期货的即时价格
    yield_rates = [(seq[i+1]-seq[i])/seq[i] for i in range(len(seq)-1)]
    return np.std(yield_rates) * np.sqrt(250)
