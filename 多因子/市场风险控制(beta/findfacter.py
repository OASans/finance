# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:49:18 2019

@author: wbl19
"""

import tushare as ts
import pandas

ts.set_token('a93f250e15311901b51e097c305d0c14d1961dd5113fa09d430b2e6b')
pro = ts.pro_api()


def get_concepts():
    #ts概念词
    df = pro.concept()
    return df

def get_stocks_inconcept(concept_code):
    df = pro.concept_detail(id=concept_code, fields='ts_code,name')
    return df

def get_classes():
    #申万行业分类
    df1 = pro.index_classify(level='L1', src='SW')
    df2 = pro.index_classify(level='L2', src='SW')
    df3 = pro.index_classify(level='L3', src='SW')
    #有三层的
    return df1,df2,df3

def get_value(stock,day):
    df = pro.daily_basic(ts_code=stock, trade_date=day, fields='ts_code,trade_date,total_mv')
    return df.total_mv[0]
    

conce=get_concepts()
stocks=get_stocks_inconcept(conce.code[0])
value=get_value(stocks.ts_code[0],"20190821")