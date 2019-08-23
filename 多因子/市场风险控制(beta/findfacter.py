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
    #ts概念词
    df = pro.concept()
    return df

def get_classes():
    #申万行业分类
    df1 = pro.index_classify(level='L1', src='SW')
    df2 = pro.index_classify(level='L2', src='SW')
    df3 = pro.index_classify(level='L3', src='SW')
    #有三层的
    return df1,df2,df3

def factor_cal(,)

