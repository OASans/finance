import tushare as ts
import pandas as pd
import numpy as np
import talib

def get_least_position(user_id: str, stock_code: str) -> int:
    '''
        返回用户某支股票的最少持仓数。
        获取方式由软件组决定，这里是一个mock。
    '''
    MOCK_NUMBER = 500
    return MOCK_NUMBER

def cal_volatility(price: list) -> float:
    '''
        计算给定序列的波动率。
        计算方式由系统其它模块决定，这里是一个mock。
    '''
    MOCK_VOL = 0.5
    return MOCK_VOL

def pred_volatility(price: list) -> float:
    '''
        预测给定序列的平均收益率。
        注意：也许这个数值会为每个股票或期货计算一次并缓存，
            因此接受的输入可能需要根据实际需要修改。
        返回一个mock。
    '''
    MOCK_VOL = 0.5
    return MOCK_VOL

def get_volatility(stock_code: str, time: int) -> float:
    '''
        从系统缓存好的数据中直接获取以time为计算频率，代码为stock_code的股票的年化波动率
    '''
    MOCK_VOL = 0.5
    return MOCK_VOL

def cal_mean_return(price: list) -> float:
    '''
        计算给定序列的平均收益率。
        计算方式由系统其它模块决定，这里是一个mock。
    '''
    MOCK_VOL = 0.5
    return MOCK_VOL

def pred_mean_return(price: list) -> float:
    '''
        预测给定序列的平均收益率。
        注意：也许这个数值会为每个股票或期货计算一次并缓存，
            因此接受的输入可能需要根据实际需要修改。
        返回一个mock。
    '''
    MOCK_VOL = 0.5
    return MOCK_VOL

def get_mean_return(stock_code: str, time: int) -> float:
    '''
        从系统缓存好的数据中直接获取以time为计算频率，代码为stock_code的股票的年化收益率
    '''
    MOCK_VOL = 0.5
    return MOCK_VOL

def get_stock_info(ts_code: str, time: int) -> pd.DataFrame:
    '''
        获取股票前time个交易日的相关信息。
        这里用tushare做一个样例，到时候软件组可以修改本函数以调用系统内部的缓存数据。
    '''
    pro = ts.pro_api()
    data_df = pro.daily(ts_code = ts_code, start_date = '20190701', end_date = '20190930')
    return data_df.iloc[:time]

def get_stock_basic(ts_code: str, time: int) -> pd.DataFrame:
    '''
        获取股票前time个交易日的指标。
        这里用tushare做一个样例，到时候软件组可以修改本函数以调用系统内部的缓存数据。
    '''
    pro = ts.pro_api()
    data_df = pro.daily_basic(ts_code = ts_code, start_date = '20190701', end_date = '20190930')
    return data_df.iloc[:time]
####以上是模拟的临时获取函数，返回的用于调试的mock数值，需要根据系统实际实现来修改################

def stock_least_position(user_id: str, stock_code: str) -> int:
    '''
        返回用户某支股票的最少持仓数，数据来自系统内部。
        
        :param user_id: 用户的身份代码，由软件组决定格式
        :param stock_code: 股票代码
        :returns: 用户某支股票的最少持仓数
    '''
    return get_least_position(user_id, stock_code)

def stock_volatility(stock_code:str, time:int) -> float:
    '''
        返回该股票在指定时间周期下的波动率。
        
        :param stock_code: 股票代码
        :param  time: 时间周期
        :returns: 波动率
    '''
    data = get_stock_info(stock_code, time)
    return cal_volatility(data['close'])
    
def stock_change(stock_code:str, time:int) -> float:
    '''
        返回该股票在指定时间周期下的涨跌幅。
        
        :param stock_code: 股票代码
        :param  time: 时间周期
        :returns: 涨跌幅
    '''
    data = get_stock_info(stock_code, time)
    cls = data['close'][0]
    pre_cls = data['close'][time - 1]
    return (cls - pre_cls) / pre_cls

def stock_mean_return(stock_code:str, time:int) -> float:
    '''
        返回该股票在指定时间周期下的平均收益率。
        
        :param stock_code: 股票代码
        :param  time: 时间周期
        :returns: 平均收益率
    '''
    data = get_stock_info(stock_code, time)
    return cal_mean_return(data['close'])

def stock_turnover_rate(stock_code: str) -> float:
    '''
        返回该股票最近一日的换手率。
        
        :param stock_code: 股票代码
        :returns: 换手率
    '''
    data = get_stock_info(stock_code, 1) # 需要当天的换手率
    return data['turnover_rate'][0]
    
def stock_macd(stock_code: str) -> float:
    '''
        返回该股票的macd指标。
        
        :param stock_code: 股票代码
        :returns: 最近一天的macd
    '''
    data = get_stock_info(stock_code, 27)
    close = [float(x) for x in data['close']]
    ret = talib.MACD(np.array(close), fastperiod=6, slowperiod=12, signalperiod=9)[0][-1] # 最近一天的MACD
    return ret

def stock_rsi(stock_code: str) -> dict:
    '''
        返回该股票的rsi指标。
        
        :param stock_code: 股票代码
        :returns: 三种rsi
    '''
    data = get_stock_info(stock_code, 25)
    close = [float(x) for x in data['close']]
    ret = {}
    ret['rsi6'] = talib.RSI(np.array(close), timeperiod=6)[-1]  # 6日rsi
    ret['rsi12'] = talib.RSI(np.array(close), timeperiod=12)[-1]  # 12日rsi
    ret['rsi24'] = talib.RSI(np.array(close), timeperiod=24)[-1]  # 24日rsi
    return ret
    
def stock_kdj(stock_code: str) -> dict:
    '''
        返回该股票的kdj指标。
        代码改自：http://www.imooc.com/article/285909
        
        :param stock_code: 股票代码
        :returns: k,d,j三个指标
    '''
    data = get_stock_info(stock_code, 9)
    ret = {}
    low_list = data['low'].rolling(9, min_periods=1).min()
    high_list = data['high'].rolling(9, min_periods=1).max()
    rsv = (data['close'] - low_list) / (high_list - low_list) * 100
    data['K'] = rsv.ewm(com=2, adjust=False).mean()
    data['D'] = data['K'].ewm(com=2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']
    ret['K'] = data['K'][0]
    ret['D'] = data['D'][0]
    ret['J'] = data['J'][0]
    return ret

def stock_roc(stock_code: str, N: int) -> float:
    '''
        返回该股票的roc指标。
        
        :param stock_code: 股票代码
        :returns: roc指标
    '''
    return stock_change(stock_code, time = N)

def stock_sharpe(stock_code: str) -> float:
    '''
        返回该股票的sharpe ratio。
        以日为单位计算。
        
        :param stock_code: 股票代码
        :returns: sharpe ratio
    '''    
    # 获取日收益率、波动率
    mrt = get_mean_return(stock_code, 1)
    vol = get_volatility(stock_code, 1)
    return mrt / vol
    
def portfolio_var(price: list) -> float:
    '''
        返回投资组合在指定时间周期下的VaR值。
        
        :param price: 投资组合的历史价值
        :returns: VaR值
    '''
    mrt = pred_mean_return(price)
    vol = pred_volatility(price)
    var_param = 1.64 # 标准正态分布的上0.95分位点
    return var_param * vol - mrt 

def portfolio_volatility(price: list) -> float:
    '''
        返回投资组合当前的波动率。
        
        :param price: 投资组合的历史价值
        :returns: 投资组合当前的波动率
    '''
    return cal_volatility(price)