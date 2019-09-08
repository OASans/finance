# -*- coding: utf-8 -*-
# author Yuan Manjie
# Date 2019.8.26

import pandas as pd
import numpy as np
import copy

stock_path = r'../获取资产的基本数据/股票/'
options_path = r'../获取资产的基本数据/期权/'
futures_path = r'../获取资产的基本数据/期货/'

contract_unit=dict()
contract_unit['50ETF']=10000
contract_unit['IF']=300
contract_unit['IC']=200
contract_unit['IH']=300

# 获取根据id获取一段时期内证券的数据


def get_stock_data(id, begin_t, end_t):
    try:
        data = pd.read_excel(stock_path + id + r'.xlsx',index_col=0)
        data=data[begin_t:end_t]
        return data.iloc[:,:1]
    except :
        return pd.DataFrame(None)


def get_futures_data(id, begin_t, end_t):
    try:
        data = pd.read_excel(futures_path + id + r'.xlsx',index_col=0)
        # data = pd.read_csv(futures_path + id + r'.csv',index_col=0,engine='python')
        data=data[begin_t:end_t]
        return data['OPEN']
    except :
        return pd.DataFrame(None)



def get_options_data(id, begin_t, end_t):
    try:
        data = pd.read_excel(options_path + id + r'.xlsx',index_col=0)
        # data = pd.read_csv(options_path + id + r'.csv',index_col=0,engine='python')
        data=data[begin_t:end_t]
        return data['OPEN']
    except :
        return pd.DataFrame(None)


def is_stock(id):
    if id[-3:] in ['.SZ']:
        return True
    else:
        return False

def is_futures(id):
    if id[:2] in ['IF']:
        return True
    else:
        return False

def is_options(id):
    if id[-2:] in ['SH'] and id[-3]!='.':
        return True
    else:
        return False

#根据旧持仓情况模拟交易计算新的持仓
#同时判断合法性,现金不够、股票100整数倍等——统一四舍五入、目前不包含手续费计算？
def cal_cash(new_p,position,cash,asset_data,asset_id,asset_data_before):
    # new_new_p=[0]*len(position)
    new_new_p=copy.deepcopy(position)
    new_c=cash
    for ii,i in enumerate(new_p):
        if i<0 and (is_stock(asset_id[ii]) or is_options(asset_id[ii])):  #暂时不许卖空 只有期货可以
            new_p[ii]=0
    for ii,i in enumerate(new_p):
        if is_stock(asset_id[ii]):
            new_p[ii]=(i+50)//100*100
        else:
            if i>0:
                new_p[ii]=int(i+0.5)
            else:
                new_p[ii]=int(i-0.5)
    is_sell=np.array(new_p)<=np.array(position)
    is_buy=np.array(new_p)>np.array(position)
    delta_p=0

    for i in range(len(new_p)):
        if is_sell[i]:
            if is_stock(asset_id[i]):
                new_c+=asset_data[i]*(position[i]-new_p[i])
                new_new_p[i]=new_p[i]
            elif is_options(asset_id[i]):
                new_c+=asset_data[i]*(position[i]-new_p[i])*contract_unit['50ETF']
                new_new_p[i]=new_p[i]
            elif is_futures(asset_id[i]):
                delta_p=(asset_data[i]-asset_data_before[i])*contract_unit[asset_id[i][:2]]*position[i]
                delta_p+=asset_data_before[i]*position[i]*contract_unit[asset_id[i][:2]]*0.08
                new_c+=delta_p
                bao=asset_data[i]*new_p[i]*contract_unit[asset_id[i][:2]]*0.08
                if new_c>=bao:
                    new_c-=bao
                    new_new_p[i]=new_p[i]
                else:#强制平仓
                    new_new_p[i]=0
    for i in range(len(new_p)):
        if is_buy[i]:
            if is_stock(asset_id[ii]):
                cash_need=asset_data[i]*(new_p[i]-position[i])
                if cash_need<=new_c:
                    new_c-=cash_need
                    new_new_p[i]=new_p[i]
                else:
                    amt_limit=new_c//asset_data[i]
                    amt_limit=amt_limit//100*100
                    new_new_p[i]=position[i]+amt_limit
                    new_c-=amt_limit*asset_data[i]
            elif is_options(asset_id[ii]):
                cash_need=asset_data[i]*(new_p[i]-position[i])*contract_unit['50ETF']
                if cash_need<=new_c:
                    new_c-=cash_need
                    new_new_p[i]=new_p[i]
                else:
                    amt_limit=new_c//(asset_data[i]*contract_unit['50ETF'])
                    new_new_p[i]=position[i]+amt_limit
                    new_c-=amt_limit*asset_data[i]*contract_unit['50ETF']
            elif is_futures(asset_id[ii]):
                delta_p=(asset_data[i]-asset_data_before[i])*contract_unit[asset_id[i][:2]]*position[i]
                delta_p+=asset_data_before[i]*position[i]*contract_unit[asset_id[i][:2]]*0.08
                new_c+=delta_p
                bao=asset_data[i]*position[i]*contract_unit[asset_id[i][:2]]*0.08
                if new_c>bao:
                    new_c-=bao
                    new_new_p[i]=position[i]
                    cash_need=asset_data[i]*(new_p[i]-position[i])*contract_unit[asset_id[i][:2]]*0.08
                    if cash_need<=new_c:
                        new_c-=cash_need
                        new_new_p[i]=new_p[i]
                    else:
                        amt_limit=new_c//(asset_data[i]*contract_unit[asset_id[i][:2]]*0.08)
                        new_new_p[i]=position[i]+amt_limit
                        new_c-=amt_limit*asset_data[i]*contract_unit[asset_id[i][:2]]*0.08
                else:#强制平仓
                    new_new_p[i]=0
    # print(is_sell,cash,new_c,delta_p)
    # print(new_new_p)
    return new_new_p,new_c


# 回测策略

#返回position/cash调整
# asset_dat为从回测开始至这一日之前的dataframe,每一列为一资产
# asset_amount为当前持仓情况,为list的list,即二维数组
# cash为当前剩余现金金额
# 返回一个新的持仓情况，即一个持有资产情况的list
# 以下只是一个例子
def policy_example1(asset_dat, asset_amount,cash):
    is_rise=asset_dat.iloc[-2]<=asset_dat.iloc[-1]
    new_p=[]
    for ii,i in enumerate(asset_amount):
        if is_rise[ii]:
            new_p+=[i+1000]
        else:
            new_p+=[i-1000]
    return new_p

def policy_example2(asset_dat,asset_amount,cash):
    return asset_amount

def policy_example3(asset_dat,asset_amount,cash):
    if asset_amount[1]==0:
        asset_amount[1]=-1
    return asset_amount

# 回测函数

# begin_asset_id 为id的list ,如['000001.SZ','000002.SZ']
#begin_t、end_t 为 str类型时间戳，如'2019-8-1'、'2019-08-01'、'2019-8'
#delta_t为整型 触发回测的天数
#policy 为策略函数
#以开盘价为模拟买入卖出价
#对于错误的asset_id,目前是直接连同持仓一起扔掉
def back_test(begin_asset_id, begin_asset_amount,begin_cash, policy, begin_t, end_t,delta_t):
    asset_data=[]
    asset_keys=[]
    asset_amount=[]
    for ii,i in enumerate(begin_asset_id):
        if is_stock(i):
            temp=get_stock_data(i,pd.Timestamp(begin_t),pd.Timestamp(end_t))
            if len(temp)==0:
                pass
            else:
                asset_data+=[temp]
                asset_keys+=[i]
                asset_amount+=[begin_asset_amount[ii]]
        elif is_options(i):
            temp=get_options_data(i,pd.Timestamp(begin_t),pd.Timestamp(end_t))
            if len(temp)==0:
                pass
            else:
                asset_data+=[temp]
                asset_keys+=[i]
                asset_amount+=[begin_asset_amount[ii]]
        elif is_futures(i):
            temp=get_futures_data(i,pd.Timestamp(begin_t),pd.Timestamp(end_t))
            if len(temp)==0:
                pass
            else:
                asset_data+=[temp]
                asset_keys+=[i]
                asset_amount+=[begin_asset_amount[ii]]

    if len(asset_data)==0:
        return pd.DataFrame()
    else:
        asset_data=pd.concat(asset_data,axis=1,keys=asset_keys)


    cashes=[begin_cash]
    positions=[asset_amount]
    last_day=pd.Timestamp(begin_t)
    for ii,i in enumerate(asset_data.index[1:]):
        if (i-last_day).days>=delta_t:
            last_day=i
            new_p=policy(asset_data.loc[:i],copy.deepcopy(positions[-1]),cashes[-1])
            new_new_p,new_c=cal_cash(new_p,positions[-1],cashes[-1],asset_data.loc[i],[x[0] for x in asset_data.columns],asset_data.loc[asset_data.index[ii-1]])
            cashes+=[new_c]
            positions+=[new_new_p]
        else:
            cashes+=[cashes[-1]]
            positions+=[positions[-1]]

    #计算总收益
    total_temp=[]
    for ii,i in enumerate(asset_data.index):
        res_sum=cashes[ii]
        for jj,j in enumerate([x[0] for x in asset_data.columns]):
            if is_stock(j):
                res_sum+=asset_data.iloc[ii,jj]*positions[ii][jj]
            elif is_futures(j):
                res_sum+=asset_data.iloc[ii,jj]*positions[ii][jj]*contract_unit[j[:2]]*0.08
            elif is_options(j):
                res_sum+=asset_data.iloc[ii,jj]*positions[ii][jj]*contract_unit['50ETF']
        total_temp+=[res_sum]
    asset_data['total']=total_temp
    return asset_data['total']


# use examples
d=back_test(['000001.SZ','000010.SZ'],[100000,100000],1000000,policy_example2,'2019-4','2019-7',1)
from matplotlib import pyplot as plt
plt.figure()
plt.plot_date(d.index,d.values,label='1',fmt='-')
print('-----------------')
dd=back_test(['000001.SZ','IF1909','000010.SZ'],[100000,0,100000],1000000,policy_example3,'2019-4','2019-7',1)
plt.plot_date(dd.index,dd.values,label='2',fmt='-')
plt.legend()
plt.show()
