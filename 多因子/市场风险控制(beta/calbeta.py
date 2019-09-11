# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:11:26 2019

@author: wbl19
"""


import sqlite3
import pandas as pd
import splitconcept
import calfacter

conn = sqlite3.connect(r'../../获取资产的基本数据/fin_set.db')#连接到db
c = conn.cursor()#创建游标


all_facter=splitconcept.get_all_facter()


stockinfo=c.execute('SELECT TRADECODE,CONCEPT,IND_NAME FROM STOCKINFO')
p=stockinfo.fetchall()


market_df=get_market_info(['DATE','RF','RM'])

for each_stock in p:
    #整理参数
    factor_list=''
    if each_stock[1] is None:
        factor_list=each_stock[2]
    else:
        factor_list=';'.join([each_stock[1],each_stock[2]])
    



