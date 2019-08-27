# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:22:21 2019

@author: yangfan
"""
from math import log,exp,sqrt
from scipy import stats


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
    d1 = (log(st/k)+(r+1/2*sigma)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    call = st*stats.norm.cdf(d1)-k*exp(-r*T)*stats.norm.cdf(d2)
    return call

def put(st,k,r,T,sigma):#看跌期权
    '''
    st,k,r,T,sigma(T以年为单位，天数应该除以365)
    '''
    d1 = (log(st/k)+(r+1/2*sigma)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    put = k*exp(-r*T)*stats.norm.cdf(-1*d2)-1*st*stats.norm.cdf(-1*d1)
    return put

#Delta: call:delta = N(d1);put:delta = N(-d1)
def delta(st,k,r,T,sigma,n=1):
    '''
    n默认为1看涨期权的delta
    n为-1为看跌期权的delta
    '''
    d1 = (log(st/k)+(r+1/2*sigma)*T)/(sigma*sqrt(T))
    delta = n*stats.norm.cdf(n*d1)
    return delta

#Gamma: gamma = N＇(d1)/(st*sigma*sqrt(T))
def gamma(st,k,r,T,sigma):
    d1 = (log(st/k)+(r+1/2*sigma)*T)/(sigma*sqrt(T))
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
    d1 = (log(st/k)+(r+1/2*sigma)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    theta = -1*(st*stats.norm.pdf(d1)*sigma)/(2*sqrt(T))-n*r*k*exp(-r*T)*stats.norm.cdf(n*d2)
    return theta

#Vega（波动率）
#vega = st*sqrt(T)*N＇(d1)
def vega(st,k,r,T,sigma):
    d1 = (log(st/k)+(r+1/2*sigma)*T)/(sigma*sqrt(T))
    vega = st*sqrt(T)*stats.norm.pdf(d1)
    return vega
