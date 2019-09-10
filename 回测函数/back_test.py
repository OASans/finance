# -*- coding: utf-8 -*-
# author Yuan Manjie
# Date 2019.8.26


import pandas as pd
import numpy as np
import copy

# stock_path = r'../获取资产的基本数据/股票/'
# options_path = r'../获取资产的基本数据/期权/'
# futures_path = r'../获取资产的基本数据/期货/'

contract_unit=dict()
contract_unit['50ETF']=10000
contract_unit['IF']=300
contract_unit['IC']=200
contract_unit['IH']=300

# 获取根据id获取一段时期内证券的数据


from options import get_options_data,get_futures_data,get_stock_data,is_options,is_futures,is_stock
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
def policy_stay_calm(asset_dat,asset_amount,cash,t1,t2,new):
    return asset_amount

from options import fit_delta,cal_option_amt,portfolio_total_value,retrain_delta_model,retrain_gamma_model,fit_gamma,retrain_beta_model,fit_beta,cal_future_amt

def policy_delta(asset_dat,asset_amount,cash,t1,t2,new):
    if new:
        retrain_delta_model('test', asset_dat.columns[:-1],asset_amount[:-1],cash,asset_dat.columns[-1])
        policy_delta.temp=fit_delta('test',asset_dat.columns[:-1],asset_amount[:-1],cash,asset_dat.columns[-1],t1,t2)
        _,policy_delta.temp2=portfolio_total_value(asset_dat.columns[:-1],asset_amount[:-1],cash, t1,t2)
        policy_delta.dat=pd.concat([policy_delta.temp,policy_delta.temp2],axis=1,keys=['delta','value'])
        policy_delta.dat=policy_delta.dat.fillna(0)

    asset_amount[-1]=cal_option_amt(policy_delta.dat.iloc[len(asset_dat)-2,1], asset_dat.columns[-1], policy_delta.dat.iloc[len(asset_dat)-2,0],str(policy_delta.dat.index[len(asset_dat)-2])[:10])
    if asset_amount[-1]>=5000:
        asset_amount[-1]=5000
    return  asset_amount

def policy_gamma(asset_dat,asset_amount,cash,t1,t2,new):
    if new:
        retrain_gamma_model('test', asset_dat.columns[:-2],asset_amount[:-2],cash,asset_dat.columns[-2],asset_dat.columns[-1])
        policy_gamma.temp=fit_gamma('test',asset_dat.columns[:-2],asset_amount[:-2],cash,asset_dat.columns[-2],asset_dat.columns[-1],t1,t2)
        _,policy_gamma.temp2=portfolio_total_value(asset_dat.columns[:-2],asset_amount[:-2],cash, t1,t2)
        policy_gamma.dat=pd.concat([policy_gamma.temp,policy_gamma.temp2],axis=1)
        policy_gamma.dat=policy_gamma.dat.fillna(0)
    asset_amount[-2]=cal_option_amt(policy_gamma.dat.iloc[len(asset_dat)-2,2], asset_dat.columns[-2], policy_gamma.dat.iloc[len(asset_dat)-2,0],str(policy_gamma.dat.index[len(asset_dat)-2])[:10])
    asset_amount[-1]=cal_option_amt(policy_gamma.dat.iloc[len(asset_dat)-2,2], asset_dat.columns[-1], policy_gamma.dat.iloc[len(asset_dat)-2,1],str(policy_gamma.dat.index[len(asset_dat)-2])[:10])
    if asset_amount[-1]>=5000:
        asset_amount[-1]=5000
    if asset_amount[-2]>=5000:
        asset_amount[-2]=5000
    return  asset_amount

def policy_beta(asset_dat,asset_amount,cash,t1,t2,new):
    if new:
        retrain_beta_model('test', asset_dat.columns[:-1],asset_amount[:-1],cash,asset_dat.columns[-1])
        policy_beta.temp=fit_beta('test',asset_dat.columns[:-1],asset_amount[:-1],cash,asset_dat.columns[-1])
        _,policy_beta.dat=portfolio_total_value(asset_dat.columns[:-1],asset_amount[:-1],cash, t1,t2)

    asset_amount[-1]=cal_future_amt(policy_beta.dat[len(asset_dat)-2], asset_dat.columns[-1], policy_beta.temp,str(policy_beta.dat.index[len(asset_dat)-2])[:10])
    return  asset_amount


def policy_example3(asset_dat,asset_amount,cash,t1,t2,new):
    if asset_amount[2]==0:
        asset_amount[2]=5
    return asset_amount

# 回测函数

# begin_asset_id 为id的list ,如['000001.SZ','000002.SZ']
#begin_t、end_t 为 str类型时间戳，如'2019-8-1'、'2019-08-01'、'2019-8'
#delta_t为整型 触发回测的天数
#policy 为策略函数
#以开盘价为模拟买入卖出价
#对于错误的asset_id,目前是直接连同持仓一起扔掉
def back_test(begin_asset_id, begin_asset_amount,begin_cash, policy, begin_t, end_t,delta_t=1):
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
            temp=get_options_data(i,pd.Timestamp(begin_t),pd.Timestamp(end_t))['OPEN']
            if len(temp)==0:
                pass
            else:
                asset_data+=[temp]
                asset_keys+=[i]
                asset_amount+=[begin_asset_amount[ii]]
        elif is_futures(i):
            temp=get_futures_data(i,pd.Timestamp(begin_t),pd.Timestamp(end_t))['OPEN']
            if len(temp)==0:
                pass
            else:
                asset_data+=[temp]
                asset_keys+=[i]
                asset_amount+=[begin_asset_amount[ii]]
    if len(asset_data)==0:
        return pd.DataFrame()
    else:
        asset_data=pd.concat(asset_data,axis=1)
        asset_data.columns=asset_keys
        asset_data=asset_data.fillna(method='ffill')
        asset_data=asset_data.fillna(method='bfill')


    cashes=[begin_cash]
    positions=[asset_amount]
    last_day=pd.Timestamp(begin_t)
    new=1
    for ii,i in enumerate(asset_data.index[1:]):
        if (i-last_day).days>=delta_t:
            last_day=i
            new_p=policy(asset_data.loc[:i],copy.deepcopy(positions[-1]),cashes[-1],asset_data.index[1],asset_data.index[-1],new)
            new=0
            new_new_p,new_c=cal_cash(new_p,positions[-1],cashes[-1],asset_data.loc[i],asset_data.columns,asset_data.loc[asset_data.index[ii-1]])
            cashes+=[new_c]
            positions+=[new_new_p]
        else:
            cashes+=[cashes[-1]]
            positions+=[positions[-1]]
    #计算总收益
    total_temp=[]
    for ii,i in enumerate(asset_data.index):
        res_sum=cashes[ii]
        for jj,j in enumerate(asset_data.columns):
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
# d=back_test(['000001.SZ','10001677.SH','10001686.SH'],[100000,0,0],100000,policy_stay_calm,'2019-1','2019-9',1)
# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot_date(d.index,d.values,label='No Hedging',fmt='-')
#
# dd=back_test(['000001.SZ','10001686.SH','10001677.SH'],[100000,0,0],100000,policy_delta,'2019-1','2019-9',1)#10001677SH
#
# # dd=back_test(['000001.SZ','IF1909','000010.SZ'],[100000,0,100000],1000000,policy_example3,'2019-4','2019-7',1)
# plt.plot_date(dd.index,dd.values,label='ML-Delta Dynamic Hedging',fmt='-')
# print('-----------------')
#
# ddd=back_test(['000001.SZ','10001686.SH','10001677.SH'],[100000,0,0],100000,policy_gamma,'2019-1','2019-9',1)
# plt.plot_date(ddd.index,ddd.values,label='Beta Hedging',fmt='-')
# plt.legend()
# plt.show()
