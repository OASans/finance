# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:55:12 2019

@author: yangfan
"""
from math import log,exp,sqrt
from scipy import stats
import sqlite3
import pandas as pd
import numpy as np
'''
公式中字母的含义：（即需要的期权数据）
st--时点t的标的物价格水平
k--期权的行权价格
r-- 无风险利率
T--期权到期日
sigma--标的物固定波动率（期权收益的标准差）
'''
#B-S期权定价模型
def call(st,k,r,T,sigma):#看涨期权
    '''
    st,k,r,T,sigma(T以年为单位，天数应该除以365)
    '''
    d1 = (log(st/k)+(r+1*sigma*sigma/2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    call = st*stats.norm.cdf(d1)-k*exp(-r*T)*stats.norm.cdf(d2)
    return call

def put(st,k,r,T,sigma):#看跌期权
    '''
    st,k,r,T,sigma(T以年为单位，天数应该除以365)
    '''
    d1 = (log(st/k)+(r+1/2*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    put = k*exp(-r*T)*stats.norm.cdf(-1*d2)-1*st*stats.norm.cdf(-1*d1)
    return put

#Delta: call:delta = N(d1);put:delta = N(-d1)=N(d1)-1
def delta(st,k,r,T,sigma,n=1):
    '''
    n默认为1看涨期权的delta
    n为-1为看跌期权的delta
    '''
    d1 = (log(st/k)+(r+1/2*sigma*sigma)*T)/(sigma*sqrt(T))
    delta = n*stats.norm.cdf(n*d1)
    return delta

#Gamma: gamma = N＇(d1)/(st*sigma*sqrt(T))
def gamma(st,k,r,T,sigma):
    d1 = (log(st/k)+(r+1/2*sigma*sigma)*T)/(sigma*sqrt(T))
    gamma = stats.norm.pdf(d1)/(st*sigma*sqrt(T))
    return gamma

#Theta（时间）
#call: theta = -1*(st*N＇(d1)*sigma)/(2*sqrt(T))-r×k*exp(-r *T)*N(d2)
#put:theta = -1*(st*N＇(d1)*sigma)/(2*sqrt(T))+r×k*exp(-r *T)*N(-1*d2)
def theta(st,k,r,T,sigma,n=1):
    '''
    n默认为1看涨期权的delta
    n为-1为看跌期权的delta
    '''
    d1 = (log(st/k)+(r+1/2*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    theta = -1*(st*stats.norm.pdf(d1)*sigma)/(2*sqrt(T))-n*r*k*exp(-r*T)*stats.norm.cdf(n*d2)
    return theta

#Vega（波动率）
#vega = st*sqrt(T)*N＇(d1)
def vega(st,k,r,T,sigma):
    d1 = (log(st/k)+(r+1/2*sigma*sigma)*T)/(sigma*sqrt(T))
    vega = st*sqrt(T)*stats.norm.pdf(d1)
    return vega

#Rho
def rho(st,k,r,T,sigma,n=1):
    d1 = (log(st/k)+(r+1/2*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    rho = k*T*exp(-r*T)*stats.norm.pdf(n*d2)
    return rho

#历史波动率
def volatility(seq1,seq2):
    #yield_rates = [(seq[i+1]-seq[i])/seq[i] for i in range(len(seq)-1)]
    yield_rates = [log(seq2[i]/seq1[i]) for i in range(len(seq1))]
    return np.std(yield_rates,ddof=1) * np.sqrt(250)

#计算希腊字母的过程
def calculation(options):#options是期权的交易代码
    #连接数据库db
    conn = sqlite3.connect('fin_set.db')#连接到db
    c = conn.cursor()#创建游标
    #将optioninfo转化为dataframe
    sql="select * from OPTIONINFO "
    sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','TRADECODE','EXE_PRICE','EXE_MODE','FIRST_DATE','LAST_DATE'])
    sql="select DATE from "+sql_dat['TRADECODE'][0][-2:]+sql_dat['TRADECODE'][0][:8]
    today_temp=list(c.execute(sql))[-1][0]
    
    today=pd.Timestamp(today_temp)#精确到时分秒
    #today_temp精确到日期
    sql_dat['LAST_DATE']=sql_dat['LAST_DATE'].map(pd.Timestamp)
    sql_dat['days_left']=sql_dat['LAST_DATE']-today
    sql_dat['days_left']=sql_dat['days_left'].map(lambda x: int(x.days))


    #MARKET
    sql1="select DATE, RF from MARKET "
    sql_mar=pd.DataFrame(list(c.execute(sql1)),columns=['DATE','RF'])
    
    #定位到今日的市场宏观数据
    rf_index=sql_mar.loc[sql_mar['DATE']==today_temp]
    #r=rf_index['RF'] #当日的无风险利率rf
    r=2.597*0.01 #test用，当时无0909的rf值
    #test：option='10001688.SH'#输入的期权代码
    
    option_list=sql_dat.loc[sql_dat['TRADECODE']==options]#该期权的行权信息和日期
    option_trans='SH'+option[:8]
    sql2="select * from "+option_trans
    sql_price=pd.DataFrame(list(c.execute(sql2)),columns=['index','DATE','PRE_SETTLE','PRE_CLOSE',
                       'OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','VWAP','OI_CHG','DELTA','GAMMA',
                       'VEGA','THETA','RHO','VOLATILITYRATIO','IMPLIEDVOL'])
    price_index=sql_price.loc[sql_price['DATE']==today_temp]
    #获取标的物即时价格水平st
    close=price_index['CLOSE']
    
    #获取期初、期末的价格--用于计算历史波动率
    vol_index1=sql_price['OPEN']
    vol_index2=sql_price['CLOSE']
    
    st=float(close)#标的物价格水平
    k=float(option_list['EXE_PRICE'])#行权价格
    T=float(option_list['days_left']/365)#到期日
    vol=volatility(vol_index1,vol_index2)#历史波动率
    sigma=vol#波动率
    mode=str(option_list['EXE_MODE'])#认购方式
    #test： print(st,k,T*365,mode)
    
    #判断看涨/看跌
    if '认购' in mode:
        delta_values=delta(st,k,r,T,sigma,1)
        gamma_values=gamma(st,k,r,T,sigma)
        vega_values=vega(st,k,r,T,sigma)
        theta_values=theta(st,k,r,T,sigma,1)
        rho_values=rho(st,k,r,T,sigma,1)
    else:
        delta_values=delta(st,k,r,T,sigma,-1)
        gamma_values=gamma(st,k,r,T,sigma)
        vega_values=vega(st,k,r,T,sigma)
        theta_values=theta(st,k,r,T,sigma,-1)
        rho_values=rho(st,k,r,T,sigma,-1)
        
    greek=[delta_values,gamma_values,vega_values,theta_values,rho_values,vol*100]
    #print(delta_values)        
    #print(gamma_values)
    #print(vega_values)
    #print(theta_values)
    #print(rho_values)
    #print(vol)
    return greek

#连接到数据库并获取今日时间
conn = sqlite3.connect('fin_set.db')#连接到db
c = conn.cursor()#创建游标
    #将optioninfo转化为dataframe
sql="select * from OPTIONINFO "
sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','TRADECODE','EXE_PRICE','EXE_MODE','FIRST_DATE','LAST_DATE'])
sql="select DATE from "+sql_dat['TRADECODE'][0][-2:]+sql_dat['TRADECODE'][0][:8]
today_temp=list(c.execute(sql))[-1][0]
#today=pd.Timestamp(today_temp)#精确到时分秒

#输入期权交易代码查询
option=str(input())
option_trans='SH'+option[:8]
#得到所有希腊字母指标及历史波动率（显示在网页的结果序列）
lis=calculation(option)
#print(lis)
#将希腊字母录入数据库
sql_delta="UPDATE "+option_trans+" SET DELTA= "+str(lis[0])+" WHERE DATE= "+str(today_temp)
sql_gamma="UPDATE "+option_trans+" SET GAMMA= "+str(lis[1])+" WHERE DATE= "+str(today_temp)
sql_vega="UPDATE "+option_trans+" SET VEGA= "+str(lis[2])+" WHERE DATE= "+str(today_temp)
sql_theta="UPDATE "+option_trans+" SET THETA= "+str(lis[3])+" WHERE DATE= "+str(today_temp)
sql_rho="UPDATE "+option_trans+" SET RHO= "+str(lis[4])+" WHERE DATE= "+str(today_temp)
sql_vol="UPDATE "+option_trans+" SET RHO= "+str(lis[5])+" WHERE DATE= "+str(today_temp)
#print(today_temp)
c.execute(sql_delta)
c.execute(sql_gamma)
c.execute(sql_vega)
c.execute(sql_theta)
c.execute(sql_rho)
c.execute(sql_vol)
conn.commit()
conn.close()
