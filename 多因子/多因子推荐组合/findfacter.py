# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:49:18 2019

@author: wbl19
"""

import tushare as ts
import pandas

ts.set_token('a93f250e15311901b51e097c305d0c14d1961dd5113fa09d430b2e6b')
pro = ts.pro_api()
#TODO:1. 得到行业/概念词 2. 线性回归 3. 协方差矩阵估计 4. Newey-West调整 5. 贝叶斯压缩 6.

def get_concepts():
    df = pro.concept()
    return df

def get_classes():
    df1 = pro.index_classify(level='L1', src='SW')
    df2 = pro.index_classify(level='L2', src='SW')
    df3 = pro.index_classify(level='L3', src='SW')
    return df1,df2,df3



#df=get_concepts()
df1,df2,df3=get_classes()
dfsz1 = pro.index_member(ts_code='000001.SZ')


dfind1 = pro.index_member(index_code='801192.SI')
dfind2 = pro.index_member(index_code='851911.SI')
dfind3 = pro.index_member(index_code='801780.SI')

data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

data = pro.query('stock_basic', exchange='SSE', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

df = pro.index_basic(market='OTH')


df = pro.opt_basic(exchange='SSE', fields='ts_code,name,exercise_type,list_date,delist_date')



