# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:11:26 2019

@author: wbl19
"""


import sqlite3
import pandas as pd
import splitconcept
import calfacter
from sklearn import linear_model 

conn = sqlite3.connect(r'../../获取资产的基本数据/fin_set.db')#连接到db
c = conn.cursor()#创建游标


all_facter=splitconcept.get_all_facter()


stockinfo=c.execute('SELECT TRADECODE,CONCEPT,IND_NAME FROM STOCKINFO')
p=stockinfo.fetchall()
model = linear_model.LinearRegression()

#market_df=calfacter.get_market_info(['DATE','RF','RM'])

'''
'''
#market_df=calfacter.cal_factors()
market_df

coef_df=pd.DataFrame(columns = market_df.columns)
coef_df=coef_df.drop(['DATE','RF','RM'],axis=1)
for each_stock in p:
    #调整股票名
    stock_code=each_stock[0].split('.')
    stock_code[0],stock_code[1]=stock_code[1],stock_code[0]
    stock_code=''.join(stock_code)
    try:
        stock_df=calfacter.get_stock_info(stock_code,['DATE','PCTCHG'])
    except:
        continue
    
    #整理参数
    factor_list=''
    if each_stock[1] is None:
        factor_list=each_stock[2]
    else:
        factor_list=';'.join([each_stock[1],each_stock[2]])
    factor_list=factor_list.split(';')  
    
    #计算
    stock_df.PCTCHG=stock_df.PCTCHG-market_df.RF/360
    stock_df=stock_df.dropna(axis=0,how='any')
    
      
    
    this_market_df=pd.DataFrame(columns = market_df.columns)
    this_market_df=pd.DataFrame()
    this_market_df.insert(0,'DATE',market_df.DATE)
    
    for each_factor in factor_list:
        if each_factor in all_facter:
            factor_index='F'+str(all_facter.index(each_factor)+8)
            this_market_df.insert(len(this_market_df.columns),factor_index,market_df[factor_index])

    this_market_df=this_market_df.where(this_market_df.notnull(), 0)
    stock_df=pd.merge(stock_df,this_market_df,on='DATE')
    
    
    Y=stock_df.PCTCHG
    X=stock_df
    X=X.drop(['DATE','PCTCHG'],axis=1)
    model.fit(X,Y)
    coef=model.coef_
    coef=coef[1:]
    coef_list=[]
    i=0
    for each_facter in all_facter:
        if each_facter in factor_list:
            coef_list.append(coef[i])
        else:
            coef_list.append(0)
    coef_df.loc[len(coef_df)] = coef_list
#不知道有没有用
#coef_df.replace([10,1e-6],0)
#coef_df.replace([-1e-6,-10],0)
coef_df.to_excel('coef_df.xlsx')