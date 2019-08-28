# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:49:18 2019

@author: wbl19
"""

import tushare as ts
import pandas

ts.set_token('a93f250e15311901b51e097c305d0c14d1961dd5113fa09d430b2e6b')
pro = ts.pro_api()


'''
外部接口
'''




def cal_factors(date: str):
    #算出所有的因子
    #输入：日期，如"20180821"
    #输出：因子字典，键为概念/分类code，值为因子值
    factors={}
    for conce in get_concepts().iterrows():
        stocks=get_stocks_inconcept(conce[1].code)
        #factors[conce[1].name]=get_mean_mv(stocks,date)
        factors[conce[1].code]=get_mean_mv(stocks,date)
    for clas in get_classes().iterrows():
        stocks=get_stocks_inclass(clas[1].index_code)
        #factors[clas[1].industry_name]=get_mean_mv(stocks,date)
        factors[clas[1].index_code]=get_mean_mv(stocks,date)
    return factors

    
def match_factors(stock:str,factors:dict):
    #取某个股票的因子
    #输入：股票代码，如"600489.SH";因子字典，上一个函数提供
    #输出：因子字典
    this_factors={}
    df = pro.concept_detail(ts_code = stock)
    for conce in df.code:
        this_factors[conce]=factors[conce]
    df = pro.index_member(ts_code=stock)
    for clas in df.index_code:
        this_factors[clas]=factors[clas]
    return this_factors

'''
以下为内部接口
'''


def get_concepts():
    #ts概念词
    df = pro.concept()
    return df

def get_stocks_inconcept(concept_code):
    df = pro.concept_detail(id=concept_code, fields='ts_code,name')
    return df

def get_classes():
    #申万行业分类
    df3 = pro.index_classify(level='L3', src='SW')
    #用三层的
    return df3

def get_stocks_inclass(concept_code):
    df = pro.index_member(id=concept_code)
    return df
    
def get_all_earn_rate(stocks, date: str):
    ret = {}
    
    df2 = pro.shibor(date= date)
    rf0 = df2['on']/360
    
    for i in stocks.ts_code:
        info_df = pro.daily(ts_code = i, start_date = date, end_date = date)
        if info_df.empty:
            ret[i] = 0
        else:
            ret[i]= info_df.change[0] / info_df.pre_close[0]-rf0[0]
    return ret

def get_mean_mv(stocks, date: str):
    earn_rate = get_all_earn_rate(stocks, date)
    weight_times_mv = 0
    mv = 0
    for i in stocks.ts_code:
        info_df = pro.daily_basic(ts_code = i, start_date = date, end_date = date)
        if (info_df.empty!=True):
            weight_times_mv += info_df.circ_mv[0] * earn_rate[i]
            mv += info_df.circ_mv[0]
    if (mv!=0):
        return weight_times_mv / mv
    else:
        return 0

cal_factors("20180821")
