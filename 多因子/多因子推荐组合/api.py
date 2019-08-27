import tushare as ts
from math import log
import pandas as pd
import datetime
from dateutil.relativedelta import *
import math
import re
from datetime import date, datetime, timedelta
pro = ts.pro_api('f80f16bef1bad9b0ac42056aa6343f1c6b74b1ce6e820e872f9b266d')


# 市值因子
def get_size(ts_code, fields):
    """
    :param
        ts_code: str
        fields: 'circ_mv' / 'total_mv'
    :return:
        the natural logarithm of circulate / total market value: float
    # example
        print(get_size('600230.SH', '20180726', 'circ_mv'))
    """
    day = date.today()
    trade_date = datetime.strftime(day, '%Y%m%d')
    return log((pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields=fields).iloc[0, 0]))


# 动量因子
def get_momentum(ts_code):
    """
    :param
        ts_code: str
    :return:
        the momentum factor of market: float
    # example
        print(get_momentum('000001.SZ'))
    """
    day = date.today()
    delta1 = timedelta(days=21)
    delta2 = timedelta(days=21+504)
    day_end = datetime.strftime(day - delta1, '%Y%m%d')
    day_begin = datetime.strftime(day - delta2, '%Y%m%d')
    df = pro.daily(ts_code=ts_code, start_date=day_begin, end_date=day_end).sort_values(by='trade_date', ascending=True)
    r = df['pct_chg']
    for index in range(0, len(r)):
        r[index] = log(1 + r[index] / 100)
    r0 = r.ewm(halflife=126).mean()[0]
    df2 = pro.shibor(start_date=day_begin, end_date=day_end)
    rf = df2['on']
    for index in range(0, len(rf)):
        rf[index] = log(1 + rf[index] / 100)
    rf0 = rf.ewm(halflife=126).mean()[0]
    return r0 - rf0


# 账面市值比因子
def get_book_to_market(stock_code):
    """
        get book to market value,
        which is, the reciprocal of Price-To-Book Ratio
    :param stock_code: str
        stock_code
    :return: float
        book to market value

    # example
    bm = get_book_to_market('300100')
    print(bm)
    """
    basic = ts.get_stock_basics().reset_index()
    pb_value = basic[basic['code'] == stock_code][['pb']].as_matrix()
    bm_value = 1.0 / pb_value
    return bm_value[0][0]


# %% 流动性因子
def get_liquidity(stock_code):
    """
        get_liquidity
    :param stock_code: str
        stock_code
    :return: float
        liquidity of the stock

    # example
    liq = get_liquidity('000001.SZ')
    print(liq)
    """
    date_now = datetime.date.today()
    date_one_year_ago = re.sub(r'-', '', str(date_now + relativedelta(years=-1)))
    date_one_month_ago = re.sub(r'-', '', str(date_now + relativedelta(years=0, months=-1)))
    date_three_month_ago = re.sub(r'-', '', str(date_now + relativedelta(years=0, months=-3)))
    date_today = re.sub(r'-', '', str(date_now))

    stom_df = ts.pro_bar(ts_code=stock_code, start_date=date_one_month_ago, end_date=date_today,
                         factors=['tor'])  # 过去一个月的换手率,理论上21天
    stoq_df = ts.pro_bar(ts_code=stock_code, start_date=date_three_month_ago, end_date=date_today,
                         factors=['tor'])  # 过去三个月的换手率
    stoa_df = ts.pro_bar(ts_code=stock_code, start_date=date_one_year_ago, end_date=date_today,
                         factors=['tor'])  # 过去一年的换手率
    stom = math.log(stoa_df['turnover_rate'].sum())
    stoq = math.log(stoq_df['turnover_rate'].sum() / 3.0)
    stoa = math.log(stoa_df['turnover_rate'].sum() / 12.0)

    liquidity = 0.35 * stom + 0.35 * stoq + 0.3 * stoa
    return liquidity




