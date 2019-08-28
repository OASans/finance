# -*- coding: utf-8 -*-
# author Yuan Manjie
# Date 2019.8.26

import pandas as pd
import numpy as np

stock_path = r'../获取资产的基本数据/股票/'

# 获取根据id获取一段时期内证券的数据


def get_stock_data(id, begin_t, end_t):
    try:
        data = pd.read_excel(stock_path + id + r'.xlsx')
        data=data[begin_t:end_t]
        return data.iloc[:,:1]
    except :
        return pd.DataFrame(None)


def get_futures_data(id, begin_t, end_t):
    pass


def get_options_data(id, begin_t, end_t):
    pass

#根据旧持仓情况模拟交易计算新的持仓
#同时判断合法性,现金不够、股票100整数倍等——统一四舍五入、目前不包含手续费计算？
def cal_cash(new_p,position,cash,asset_data):
    is_sell=np.array(new_p)<=np.array(position)
    new_new_p=[0]*len(position)
    new_c=cash
    for ii,i in enumerate(new_p):
        if i<0:
            new_p[ii]=0
    new_p=[(x+50)//100*100 for x in new_p]  #need to be modified to fit options and futures
    for i in range(len(new_p)):
        if is_sell[i]:
            new_c+=asset_data[i]*(position[i]-new_p[i])
            new_new_p[i]=new_p[i]
    for i in range(len(new_p)):
        if not is_sell[i]:
            cash_need=asset_data[i]*(new_p[i]-position[i])
            if cash_need<=new_c:
                new_c-=cash_need
                new_new_p[i]=new_p[i]
            else:
                amt_limit=new_c//asset_data[i]
                amt_limit=amt_limit//100*100
                new_new_p[i]=position[i]+amt_limit
                new_c-=amt_limit*asset_data[i]
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
        if i[-3:]=='.SZ':
            temp=get_stock_data(i,begin_t,end_t)
            if len(temp)==0:
                pass
            else:
                asset_data+=[temp]
                asset_keys+=[i]
                asset_amount+=[begin_asset_amount[ii]]
        else:
            pass
    asset_data=pd.concat(asset_data,axis=1,keys=asset_keys)

    cashes=[begin_cash]
    positions=[asset_amount]
    last_day=pd.Timestamp(begin_t)
    for i in asset_data.index[1:]:
        if (i-last_day).days>=delta_t:
            last_day=i
            new_p=policy(asset_data.loc[:i],positions[-1],cashes[-1])
            new_new_p,new_c=cal_cash(new_p,positions[-1],cashes[-1],asset_data.loc[i])
            cashes+=[new_c]
            positions+=[new_new_p]
        else:
            cashes+=[cashes[-1]]
            positions+=[positions[-1]]

    #计算总收益
    total_temp=[]
    for ii,i in enumerate(asset_data.index):
        total_temp+=[np.sum(np.array(asset_data.loc[i])*positions[ii])+cashes[ii]]
    asset_data['total']=total_temp
    return asset_data['total']



# use examples
# d=back_test(['000001.SZ','000003.SZ','000010.SZ'],[1000,1000,1000],1000000,policy_example1,'2018-9','2019-7',1)
# from matplotlib import pyplot as plt
# plt.plot_date(d.index,d.values,fmt='-')
# d=back_test(['000001.SZ','000003.SZ','000010.SZ'],[1000,1000,1000],1000000,policy_example2,'2018-9','2019-7',1)
# plt.plot_date(d.index,d.values,fmt='-')
