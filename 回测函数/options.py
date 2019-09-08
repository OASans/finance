# -*- coding: utf-8 -*-
# author Yuan Manjie
# date 2019/9/5

import pandas as pd
import numpy as np


contract_unit=dict()
contract_unit['50ETF']=10000
contract_unit['IF']=300
contract_unit['IC']=200
contract_unit['IH']=300

stock_path = r'../获取资产的基本数据/股票/'
options_path = r'../获取资产的基本数据/期权/'
futures_path = r'../获取资产的基本数据/期货/'

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
        return data
    except :
        return pd.DataFrame(None)

def get_options_data(id, begin_t, end_t):
    try:
        data = pd.read_excel(options_path + id + r'.xlsx',index_col=0)
        # data = pd.read_csv(options_path + id + r'.csv',index_col=0,engine='python')
        data=data[begin_t:end_t]
        return data
    except :
        return pd.DataFrame(None)

def is_stock(id):
    if id[-3:] in ['.SZ']:
        return True
    else:
        return False

def is_futures(id):
    if id[:2] in ['IF','IC','IH']:
        return True
    else:
        return False

def is_options(id):
    if id[-2:] in ['SH'] and id[-3]!='.':
        return True
    else:
        return False

def portfolio_total_value(asset_id,asset_mount,cash,begin_t, end_t):
    total=[]
    stocks=[]
    keys=[]
    stock_keys=[]
    for ii,i in enumerate(asset_id):# 到期？ 保证金不足？
        if is_stock(i):
            temp=get_stock_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                stock_keys+=[i]
                total+=[temp*asset_mount[ii]]
                stocks+=[temp*asset_mount[ii]]
        elif is_futures(i):
            temp=get_futures_data(i, begin_t, end_t)
            if len(temp)!=0:
                keys+=[i]
                temp['delta']=temp['OPEN'].diff(1)
                temp=temp.fillna(0)
                temp['delta']*=contract_unit[i[:2]]
                temp['delta'][0]+=temp['OPEN'][0]*contract_unit[i[:2]]*0.08
                for j in range(1,len(temp)):
                    temp['delta'][j]+=temp['delta'][j-1]
                    if temp['delta'][j]<=0:
                        temp['delta'][j:]=0
                        break
                total+=[temp['delta']*asset_mount[ii]]
        elif is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            temp=temp.fillna(0)
            if len(temp)!=0:
                keys+=[i]
                total+=[temp['OPEN']*asset_mount[ii]*contract_unit[temp['US_NAME'][0]]]
    if len(total)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total,axis=1,keys=keys)
        if len(stocks)==0:
            stocks=pd.DataFrame()
        else:
            stocks=pd.concat(stocks,axis=1,keys=stock_keys)
    return total.sum(axis=1)+cash,stocks.sum(axis=1)

def portfolio_delta(asset_id,asset_mount,cash,begin_t, end_t):#单个值可能不好取   单位上？
    total,stock=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    delta=total.diff(1)/stock.diff(1)
    return delta[1:]

def portfolio_gamma(asset_id,asset_mount,cash,begin_t, end_t):
    total,stock=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    delta=total.diff(1)/stock.diff(1)
    gamma=delta.diff(1)/stock.diff(1)
    return gamma[2:]

def portfolio_vega(asset_id,asset_mount,cash,begin_t,end_t):
    total_vega=[]
    keys=[]
    for ii,i in enumerate(asset_id):
        if is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                total_vega+=[temp['VEGA']*asset_mount[ii]*contract_unit[temp['US_NAME'][0]]]
    if len(total_vega)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total_vega,axis=1,keys=keys)
        total=total.fillna(0)
        return total.sum(axis=1)

def portfolio_rho(asset_id,asset_mount,cash,begin_t,end_t):
    total_rho=[]
    keys=[]
    for ii,i in enumerate(asset_id):
        if is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                total_rho+=[temp['RHO']*asset_mount[ii]*contract_unit[temp['US_NAME'][0]]]
        elif is_futures(i):
            pass
            # temp=get_futures_data(i, begin_t, end_t)
            # if len(temp)!=0:
            #     keys+=[i]
            #     temp['rho']=list(map(lambda x:1.015**(x.days/365)*x.days/365,temp['LASTTRADE_DATE']-temp.index))
            #     total_rho+=[temp['OPEN']*temp['rho']*asset_mount[ii]*contract_unit[i[:2]]]
    if len(total_rho)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total_rho,axis=1,keys=keys)
        total=total.fillna(0)
        return total.sum(axis=1)

def portfolio_theta(asset_id,asset_mount,cash,begin_t,end_t):
    total_theta=[]
    keys=[]
    for ii,i in enumerate(asset_id):
        if is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                total_theta+=[temp['THETA']*asset_mount[ii]*contract_unit[temp['US_NAME'][0]]]
        elif is_futures(i):
            pass
            # temp=get_futures_data(i, begin_t, end_t)
            # if len(temp)!=0:
            #     keys+=[i]
            #     temp['theta']=list(map(lambda x:-np.log(1.015)*1.015**(x.days/365),temp['LASTTRADE_DATE']-temp.index))
            #     total_theta+=[temp['OPEN']*temp['theta']*asset_mount[ii]*contract_unit[i[:2]]]
    if len(total_theta)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total_theta,axis=1,keys=keys)
        total=total.fillna(0)
        return total.sum(axis=1)

# portfolio_var 调用这里portfolio_total_value取第一个返回值变为list调用定期调整与条件触发中的代码实现

def portfolio_volatility(asset_id,asset_mount,cash,begin_t,end_t,time):
    total,_=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    res=total.diff(1)/total
    res=res.rolling(time).var()
    return res.dropna()

def portfolio_earning_rate(asset_id,asset_mount,cash,begin_t,end_t,time):
    total,_=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    res=total.diff(1)/total
    res=res.rolling(time).mean()
    # res=res/time*365
    return res.dropna()

def cal_option_amt(total_value,option,portion):
    temp=get_options_data(option,-2,-1)
    if len(temp)<=0:
        return 0
    else:
        res=total_value*portion/contract_unit[temp['US_NAME'][0]]/temp['EXE_PRICE'][-1]
        return int(res+0.5)

def cal_future_amt(total_value,futures,portion):
    temp=get_futures_data(futures,-2,-1)
    if len(temp)<=0:
        return 0
    else:
        res=total_value*portion/contract_unit[futures[:2]]/temp['OPEN'][-1]
        return int(res+0.5)

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
def load_train_data(asset_id,asset_mount,cash,options,begin_t='',end_t='',mode=0):
    data = pd.read_excel(options_path + options + r'.xlsx',index_col=0)
    s,_=portfolio_total_value(asset_id, asset_mount, cash, data.index[0], data.index[-1])
    data=pd.concat([data,s],axis=1)
    data.columns=list(data.columns[:-1])+['s']
    data=data[~np.isnan(data['s'])]

    data['ds']=data['s'].diff()
    data['f']=data['OPEN']*contract_unit[data['US_NAME'][0]]
    data['df']=data['f'].diff()
    data['real_delta']=data['df']/data['ds']
    if mode==1:
        data['dds']=data['real_delta'].diff()
        data['real_gamma']=data['dds']/data['ds']

    if mode==1:
        data['gamma_pre']=data['GAMMA'].shift(-1)
        data['real_gamma']=data['real_gamma'].shift(-1)

    data['delta_pre']=data['DELTA'].shift(-1)
    data['real_delta']=data['real_delta'].shift(-1)#错开一天，即为预测下一天


    data['PCT_CHG']=data['df']/data['OPEN']
    data['HV_5']=(data['PCT_CHG'].rolling(window=5).std().values) #波动率
    data['HV_10']=(data['PCT_CHG'].rolling(window=10).std().values)
    data['HV_15']=(data['PCT_CHG'].rolling(window=15).std().values)
    data['HV_20']=(data['PCT_CHG'].rolling(window=20).std().values)

    data['SoverK']=data['s']/data['EXE_PRICE']
    data['T-t']=list(map(lambda x:x.days/365,data['EXE_ENDDATE']-data.index))

    for i in range(len(data)): #隐含波动率没有解的（为空的），用20日波动率替代
        if data['US_IMPLIEDVOL'][i]==0:
            data['US_IMPLIEDVOL'][i]=data['HV_20'][i]

    data.dropna(inplace=True)

    data=data[abs(data['real_delta'])<=0.95] #选取delta在正负0.95以内的数据训练，删去极端值，原因是参考某某论文
    # if mode==1:
    #     pass
    # else:
    #     data['real_delta']=1/data['real_delta']

    if mode==1:
        data_train=data[['EXE_PRICE','s','T-t','US_IMPLIEDVOL','HV_5','HV_10','HV_15','HV_20','VOLATILITYRATIO','DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO','VWAP', 'VOLUME', 'AMT', 'OI_CHG', 'SETTLE','HIGH', 'LOW','delta_pre','real_delta','gamma_pre','real_gamma']]
    else:
        data_train=data[['EXE_PRICE','s','T-t','US_IMPLIEDVOL','HV_5','HV_10','HV_15','HV_20','VOLATILITYRATIO','DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO','VWAP', 'VOLUME', 'AMT', 'OI_CHG', 'SETTLE','HIGH', 'LOW','delta_pre','real_delta']]
    data_train.dropna(inplace=True)
    if begin_t=='' or end_t=='':
        return data_train
    else:
        return data_train[begin_t:end_t]

def train_delta_model(protfolio_id,asset_id,asset_mount,cash,options,num=0):
    data_train=load_train_data(asset_id, asset_mount, cash, options)
    rfr = RandomForestRegressor()
    rfr.fit(data_train.iloc[:,:-2].values,data_train.iloc[:,-1].values)
    joblib.dump(rfr,str(protfolio_id)+"_delta"+str(num)+".m")

    # def sse(dataa,x):
    #     return np.sum(np.square(dataa['real_delta']-x))#或是real_delta
    # def test(dataa,x):
    #     p=np.array(x.predict(dataa.iloc[:,:-2].values))
    #     sse1=sse(dataa,p)
    #     sse2=sse(dataa,dataa['delta_pre'])#或是delta_pre
    #     res=1-sse1/sse2
    #     print(sse1,sse2)
    #     print(res)
    # test(data_train,rfr)

    return rfr

def train_gamma_model(protfolio_id,asset_id,asset_mount,cash,options,num):
    data_train=load_train_data(asset_id, asset_mount, cash, options,mode=1)
    rfr = RandomForestRegressor()
    rfr.fit(data_train.iloc[:,:-4].values,data_train.iloc[:,-1].values)
    joblib.dump(rfr,str(protfolio_id)+"_gamma"+str(num)+".m")

    return rfr

def fit_delta(protfolio_id,asset_id,asset_mount,cash,options,begin_t,end_t):
    try:
        model=joblib.load(str(protfolio_id)+"_delta0.m")
    except:
        model=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options)
    data=load_train_data(asset_id,asset_mount,cash,options,begin_t,end_t)
    res=model.predict(data.iloc[:,:-2].values)
    res=list(map(lambda x:0 if x<=0 else 1/x,res))
    return res

def fit_gamma(protfolio_id,asset_id,asset_mount,cash,options1,options2,begin_t,end_t):
    try:
        model1=joblib.load(str(protfolio_id)+"_gamma1.m")
    except:
        model1=train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options1,1)
    try:
        model2=joblib.load(str(protfolio_id)+"_gamma2.m")
    except:
        model2=train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options2,2)
    try:
        model3=joblib.load(str(protfolio_id)+"_delta1.m")
    except:
        model3=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options1,1)
    try:
        model4=joblib.load(str(protfolio_id)+"_delta2.m")
    except:
        model4=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options2,2)

    data1=load_train_data(asset_id,asset_mount,cash,options1,begin_t,end_t,mode=1)
    data2=load_train_data(asset_id,asset_mount,cash,options2,begin_t,end_t,mode=1)
    res1=model1.predict(data1.iloc[:,:-4].values)
    res2=model2.predict(data2.iloc[:,:-4].values)
    res3=model3.predict(data1.iloc[:,:-4].values)
    res4=model4.predict(data2.iloc[:,:-4].values)

    # res=list(map(lambda x:0 if x<=0 else 1/x,res))
    return res

def generate_recommend_option_delta(protfolio_id,asset_id,asset_mount,cash):
    pass


def portfolio_diff():
    pass

type(portfolio_volatility(['000001.SZ','000010.SZ'],[1000,1000],100000,'2019-4','2019-5',7))

a=fit_delta('123',['000001.SZ','000010.SZ'],[1000,1000],100000,'10001677SH','2019-4','2019-5')
a
portfolio_total_value(['000001.SZ','000010.SZ'],[1000,1000],100000,'2019-4','2019-5')

portfolio_total_value(['IF1909'],[10],100000,'2019-4','2019-5')
cal_option_amt(,'10001686SH',)

d=portfolio_earning_rate(['000001.SZ','10001686SH','IF1909','000010.SZ'], [1000,10,5,1000], 100000, '2019-3', '2019-7',30)
d
pd.Timestamp('2019-3')-pd.to_timedelta(1)
np.log(np.e)
data=get_futures_data('IF1909','2019-3','2019-4')
data=get_options_data('10001686SH','2019-3','2019-4')
data
list(map(lambda x:x.days/365,data['LASTTRADE_DATE']-data.index))
