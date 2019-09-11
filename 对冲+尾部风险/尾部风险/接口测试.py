# encoding: utf8
import sqlite3
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq
from scipy import sparse
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import interpolate
import sys
from scipy.misc import derivative
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy as s
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import pylab


# 从数据库获取所需数据
def getParam():
    conn = sqlite3.connect('../../获取资产的基本数据/fin_set.db')  # 连接到db
    c = conn.cursor()  # 创建游标

    optionInfo_sql = "select TRADECODE, EXE_PRICE, EXE_MODE, ENDDATE from OPTIONINFO"
    optionInfo = pd.DataFrame(list(c.execute(optionInfo_sql)), columns=['期权代码', '行权价格', '看涨/看跌', '剩余存续期'])

    # yesterday = (datetime.datetime.now()+datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
    yesterday = "2019-09-06"  # 深夜测试，数据库未更新

    for i, v in enumerate(optionInfo['剩余存续期']):
        expiredate = datetime.datetime.strptime(v, '%Y-%m-%d') - datetime.datetime.strptime(yesterday, '%Y-%m-%d')
        optionInfo.loc[i, '剩余存续期'] = expiredate.days / 365

    optionClose = []
    for i in optionInfo['期权代码']:
        codeStr = 'SH' + i[:8]
        optionClose_sql = "select CLOSE from " + codeStr + " WHERE DATE='" + yesterday + "'"
        close = list(c.execute(optionClose_sql))[0]
        optionClose.append(close)
    optionClose = pd.DataFrame(optionClose, columns=['期权收盘价'])
    optionInfo = pd.concat([optionInfo, optionClose], axis=1)
    optionInfo['红利q'] = 0

    r_sql = "select RF from MARKET WHERE DATE='" + yesterday + "'"
    r = list(c.execute(r_sql))[0][0]

    S_sql = "select CLOSE from SH510050 WHERE DATE='" + yesterday + "'"
    S = list(c.execute(S_sql))[0][0]

    conn.close()

    #optionInfo = optionInfo[optionInfo['剩余存续期'] > optionInfo['剩余存续期'].min()]
    # 返回期权信息、无风险利率、50ETF收盘价
    return optionInfo, r, S


# 分离认购认沽期权
def splitCallPut(option_data):
    data_call = option_data[option_data['看涨/看跌'] == '认购']
    data_put = option_data[option_data['看涨/看跌'] == '认沽']
    return data_call, data_put


# 计算标的资产价格(计算隐含波动率时用)
def assetPrice(c, p, K, r, t):
    S = c+K*math.exp(-r*t)-p
    return S


# 计算data_call, data_put的标的资产价格
def assetPriceCallPut(data_call, data_put, r):
    data_call.loc[:, '标的资产价格'] = 0.00
    data_put.loc[:, '标的资产价格'] = 0.00
    r = r / 100
    for i in data_call.index:
        for j in data_put.index:
            if data_call['行权价格'][i] == data_put['行权价格'][j] and data_call['剩余存续期'][i] == data_put['剩余存续期'][j]:
                c = data_call['期权收盘价'][i]
                p = data_put['期权收盘价'][j]
                K = data_call['行权价格'][i]
                t = data_call['剩余存续期'][i]
                S = assetPrice(c, p, K, r, t)
                data_call.loc[i, '标的资产价格'] = S
                data_put.loc[j, '标的资产价格'] = S
    return data_call, data_put


# 筛选虚值状态期权用以研究
def getOutofMoney(data_call, data_put):
    call_data = data_call[data_call['行权价格'] > data_call['标的资产价格']]
    put_data = data_put[data_put['行权价格'] < data_put['标的资产价格']]
    return call_data, put_data


# 虚值状态看涨期权隐含波动率
def ImpVolCall(MktPrice, Strike, Expiry, Asset, IntRate, Dividend, Sigma, error):
    n = 1
    Volatility = Sigma  # 初始值
    dv = error + 1
    while abs(dv) > error:
        d1 = np.log(Asset / Strike) + (IntRate - Dividend + 0.5 * Volatility ** 2) * Expiry
        d1 = d1 / (Volatility * np.sqrt(Expiry))
        d2 = d1 - Volatility * np.sqrt(Expiry)
        PriceError = Asset * math.exp(-Dividend * Expiry) * norm.cdf(d1) - Strike * math.exp(
            -IntRate * Expiry) * norm.cdf(d2) - MktPrice
        Vega1 = Asset * np.sqrt(Expiry / 3.1415926 / 2) * math.exp(-0.5 * d1 ** 2)
        dv = PriceError / Vega1
        Volatility = Volatility - dv
        n = n + 1
        if n > 300:
            ImpVolCall = 0.0
            break
        ImpVolCall = Volatility
    return ImpVolCall


# 虚值状态看跌期权隐含波动率
def ImpVolPut(MktPrice, Strike, Expiry, Asset, IntRate, Dividend, Sigma, error):
    n = 1
    Volatility = Sigma  # 初始值
    dv = error + 1
    while abs(dv) > error:
        d1 = np.log(Asset / Strike) + (IntRate - Dividend + 0.5 * Volatility ** 2) * Expiry
        d1 = d1 / (Volatility * np.sqrt(Expiry))
        d2 = d1 - Volatility * np.sqrt(Expiry)
        PriceError = -Asset * math.exp(-Dividend * Expiry) * norm.cdf(-d1) + Strike * math.exp(
            -IntRate * Expiry) * norm.cdf(-d2) - MktPrice
        Vega1 = Asset * np.sqrt(Expiry / 3.1415926 / 2) * math.exp(-0.5 * d1 ** 2)
        dv = PriceError / Vega1
        Volatility = Volatility - dv
        n = n + 1
        if n > 300:
            ImpVolPut = 0.0
            break
        ImpVolPut = Volatility
    return ImpVolPut


# 计算call_data, put_data各自的隐含波动率
def impVolCal(call_data, put_data, r):
    # call
    call_data.index = range(len(call_data))
    Sigma, error = 1, 0.001
    for j in range(len(call_data)):
        MktPrice = call_data.loc[j, '期权收盘价']
        Strike = call_data.loc[j, '行权价格']
        Expiry = call_data.loc[j, '剩余存续期']
        Asset = call_data.loc[j, '标的资产价格']
        IntRate = r / 100
        Dividend = call_data.loc[j, '红利q']
        volatility = ImpVolCall(MktPrice, Strike, Expiry, Asset, IntRate, Dividend, Sigma, error)
        call_data.loc[j, '隐含波动率'] = volatility
    # put
    put_data.index = range(len(put_data))
    Sigma, error = 1, 0.001
    for j in range(len(put_data)):
        MktPrice = put_data.loc[j, '期权收盘价']
        Strike = put_data.loc[j, '行权价格']
        Expiry = put_data.loc[j, '剩余存续期']
        Asset = put_data.loc[j, '标的资产价格']
        IntRate = r / 100
        Dividend = put_data.loc[j, '红利q']
        volatility = ImpVolPut(MktPrice, Strike, Expiry, Asset, IntRate, Dividend, Sigma, error)
        put_data.loc[j, '隐含波动率'] = volatility
    return call_data, put_data


# 取有效信息，拼接，按行权价格和剩余存续期（还有多久到期）排序
def getRes(call_data, put_data):
    res_df = pd.concat([put_data[['行权价格', '剩余存续期', '隐含波动率', '看涨/看跌']], call_data[['行权价格', '剩余存续期', '隐含波动率', '看涨/看跌']]])
    res_df = res_df.sort_values(['行权价格', '剩余存续期'])
    res_df.index = range(len(res_df))
    return res_df


# 将以上res_df转化为方差矩阵，在行权价方向上进行样条插值，在剩余存续期方向上进行线性插值。
# 然后将得到的方差矩阵再转化为波动率矩阵
def impVolInterp(res_df):
    # 转化为方差矩阵
    res = res_df
    res['隐含方差'] = res['隐含波动率'] ** 2 * res['剩余存续期']

    count = 0
    for i, v in enumerate(res['行权价格']):
        max_price = res.iat[-1, 0]
        while max_price == res.iat[-i-1, 0]:
            count += 1
            i += 1
        if count != 0:
            break
    dup_min = 0
    for i, v in enumerate(res['行权价格']):
        if v == res.iat[i-1+count, 0]:
            dup_min = v
            break

    spline_data = res[(res['行权价格'] <= res['行权价格'].max()) & (dup_min <= res['行权价格'])]
    vol_mat = []
    for j in list(spline_data['行权价格'].unique()):
        vol_mat.append(list(spline_data[spline_data['行权价格'] == j]['隐含方差']))
    vol_mat = pd.DataFrame(vol_mat)
    vol_after_k = pd.DataFrame([])
    # 在行权价方向上进行样条插值
    for j in range(vol_mat.shape[1]):
        k = np.array(list(spline_data['行权价格'].unique()))
        kmesh = np.linspace(k.min(), k.max(), 300)
        volinter1 = interpolate.spline(k, np.array(vol_mat[j]), kmesh)
        vol_after_k['期限' + str(j)] = volinter1
    #     tck = interpolate.splrep(k,np.array(vol_mat[j]))
    #     volinter1 = interpolate.splev(kmesh, tck)
    #     vol_after_k['期限'+str(j)] = volinter1
    vol_after_k.index = kmesh
    # 在剩余存续期方向上进行线性插值
    tt = np.array(list(res['剩余存续期'].unique()))
    tt.sort()
    tmesh = np.linspace(tt.min(), tt.max(), 300)
    res_kt = []
    for j in vol_after_k.index:
        volinter2 = np.interp(tmesh, tt, np.array(vol_after_k.loc[j, :]))
        res_kt.append(volinter2)
    vol_after_kt = pd.DataFrame(res_kt)
    vol_after_kt.index = vol_after_k.index
    vol_after_kt.columns = tmesh
    for j in vol_after_kt.index:
        vol_after_kt.loc[j, :] = np.sqrt(np.array(vol_after_kt.loc[j, :]) / tmesh)
    return vol_after_kt, tmesh, kmesh


# 看涨看跌平价转换，然后计算看涨期权定价
def callPricing(vol_after_kt, r, tmesh, kmesh, S, q=0):
    r = r/100
    call_price = vol_after_kt.copy()
    for j in tmesh:
        for i in kmesh:
            v = vol_after_kt.loc[i, j]
            d1 = np.log(S / i) + (r - q + 0.5 * v ** 2) * j
            d1 = d1 / (v * np.sqrt(j))
            d2 = d1 - v * np.sqrt(j)
            if i >= S:
                # 看涨
                call_price.loc[i, j] = S * math.exp(-q * j) * norm.cdf(d1) - i * math.exp(-r * j) * norm.cdf(d2)
            else:
                # 看跌
                #             temp_put_price = -S*math.exp(-q*j)*norm.cdf(-d1)+i*math.exp(-r*j)*norm.cdf(d2)
                #            call_price.loc[i,j] = -i*math.exp(-r*j)+temp_put_price+S
                temp_put_price = -S * math.exp(-q * j) * norm.cdf(-d1) + i * math.exp(-r * j) * norm.cdf(-d2)
                call_price.loc[i, j] = -i * math.exp(-r * j) + temp_put_price + S * math.exp(-q * j)
    return call_price


# 隐含概率分布（这里选取的是最近到期的日子）
def impProbability(call_price, S, r):
    x = np.array(call_price.index)
    t = call_price.columns[0]
    y = np.array(list(call_price.iloc[:, 0]))
    h = x[1] - x[0]
    d2 = []
    # 差分
    for i in range(1, len(y) - 1):
        d2.append(math.exp(r * t) * ((y[i - 1] + y[i + 1] - 2 * y[i]) / (h ** 2)))
    xx = x[1:len(x) - 1]
    # 可以画个概率分布图
    # plot2 = plt.plot(xx, d2, 'b', label='diff')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend(loc=3)  # 设置图示的位置
    # plt.title('density of price(2017-10-17)')  # 设置标题
    # plt.show()
    # 转化为收益率概率分布
    rp = []
    rx = []
    for i in range(len(xx)):
        RT = np.log(xx[i] / S)
        rx.append(RT)
        rp.append(d2[i] * S * math.exp(RT))
    return rx, rp


# 拟合函数
def fit_func1(k, x):
    #     k, b = a
    #     return k * x + b
    # c!=0
    a, b, c = k
    # return (((1+c*(x-a)/b)**(-1-1/c))/b)*np.exp((1+c*(x-a)/b)**(-1/c))
    # c==0
    return np.exp((a - x) / b) * np.exp(-np.exp((a - x) / b)) / b


def fit_func2(k, x):
    #     k, b = a
    #     return k * x + b
    # c!=0
    a, b, c = k
    # return (((1+c*(x-a))**(-1-1/c))/b)*np.exp(-np.exp((a-x)/b))
    # c==0
    return np.exp((a + x) / b) * np.exp(-np.exp((a + x) / b)) / b


# 残差
def dist1(k, x, y):
    return fit_func1(k, x) - y


def dist2(k, x, y):
    return fit_func2(k, x) - y


# GEV广义极值分布填充尾部
def gevTail(rx, rp):
    x_range = np.arange(len(rx))

    # 目前计算的是-6.6%~4.2%收益率之间的概率密度

    rx = np.array(rx)
    rp = np.array(rp)

    # 填充右尾巴
    x90 = int(np.percentile(x_range, 90))
    x91 = int(np.percentile(x_range, 91))
    x92 = int(np.percentile(x_range, 92))
    x93 = int(np.percentile(x_range, 93))
    x94 = int(np.percentile(x_range, 94))
    x95 = int(np.percentile(x_range, 95))
    x96 = int(np.percentile(x_range, 96))
    x97 = int(np.percentile(x_range, 97))
    x98 = int(np.percentile(x_range, 98))
    x99 = int(np.percentile(x_range, 99))
    x100 = int(np.percentile(x_range, 100))

    fit_x_right = np.array([x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100])
    fit_r_right = []
    fit_p_right = []

    for i in fit_x_right:
        fit_r_right.append(rx[i])
        fit_p_right.append(rp[i])

    fit_r_right = np.array(fit_r_right)
    fit_p_right = np.array(fit_p_right)

    # plt.figure(figsize=(15, 10))
    # plt.title(u'GEV right tail')
    # plt.xlabel(u'RT')
    # plt.ylabel(u'density')
    # plt.plot(fit_r_right, fit_p_right, 'k.')
    # plt.plot(rx, rp, 'b', label='diff')

    par = [1, 1, 0]

    var = leastsq(dist1, par, args=(fit_r_right, fit_p_right))
    a, b, c = var[0]
    # print(a, b, c)

    right_predict_x = np.linspace(rx.min(), rx.max() * 4, 800)
    right_predict_y = np.exp((a - right_predict_x) / b) * np.exp(-np.exp((a - right_predict_x) / b)) / b

    # 填充左尾巴
    x0 = int(np.percentile(x_range, 35))
    x1 = int(np.percentile(x_range, 36))
    x2 = int(np.percentile(x_range, 37))
    x3 = int(np.percentile(x_range, 38))
    x4 = int(np.percentile(x_range, 39))
    x5 = int(np.percentile(x_range, 40))
    x6 = int(np.percentile(x_range, 41))
    x7 = int(np.percentile(x_range, 42))
    x8 = int(np.percentile(x_range, 43))
    x9 = int(np.percentile(x_range, 44))
    x10 = int(np.percentile(x_range, 45))

    fit_x_left = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    fit_r_left = []
    fit_p_left = []

    for i in fit_x_left:
        fit_r_left.append(rx[i])
        fit_p_left.append(rp[i])

    fit_r_left = np.array(fit_r_left)
    fit_p_left = np.array(fit_p_left)

    # plt.figure(figsize=(15, 10))
    # plt.title(u'GEV left tail')
    # plt.xlabel(u'RT')
    # plt.ylabel(u'density')
    # plt.plot(fit_r_left, fit_p_left, 'k.')
    # plt.plot(rx, rp, 'b', label='diff')

    par = [1, 1, 1]

    var = leastsq(dist2, par, args=(fit_r_left, fit_p_left))
    a1, b1, c1 = var[0]
    #print(a1, b1, c1)

    left_predict_x = np.linspace(rx.min() * 3, rx.max(), 1000)
    left_predict_y = np.exp((a1 + left_predict_x) / b1) * np.exp(-np.exp((a1 + left_predict_x) / b1)) / b1

    # 填充尾部后的图
    # plt.figure(figsize=(20, 10))
    # plt.title(u'GEV tail')
    # plt.xlabel(u'RT')
    # plt.ylabel(u'density')
    # plt.plot(fit_r_left, fit_p_left, 'k.')
    # plt.plot(fit_r_right, fit_p_right, 'k.')
    # plt.plot(rx, rp, 'b', label='diff')
    #
    # plt.plot(right_predict_x, right_predict_y, 'c', linestyle=":")
    #
    # plt.plot(left_predict_x, left_predict_y, 'y', linestyle=":")
    #
    # plt.show()

    # 截取完整概率分布
    get_right_x = right_predict_x[right_predict_x >= rx[x100]]
    get_left_x = left_predict_x[left_predict_x <= rx[x0]]
    finish_x = np.hstack((get_left_x, rx[x1:x99]))
    finish_x = np.hstack((finish_x, get_right_x))
    finish_y = np.hstack((np.exp((a1 + get_left_x) / b1) * np.exp(-np.exp((a1 + get_left_x) / b1)) / b1, rp[x1:x99]))
    finish_y = np.hstack((finish_y, np.exp((a - get_right_x) / b) * np.exp(-np.exp((a - get_right_x) / b)) / b))

    # 至此，期权隐含概率分布施工完成
    return finish_x, finish_y


# 期权隐含概率分布所携带的信息
# 期权隐含概率分布的隐含矩：二阶矩-隐含波动率、三阶矩-隐含偏度、四阶矩-隐含峰度
def impInfo(finish_y):
    mean_p = finish_y.mean()
    temp_y = pd.Series(finish_y)

    imp_vol = temp_y.std()
    imp_skew = temp_y.skew()
    imp_kurt = temp_y.kurt()

    return imp_vol, imp_skew, imp_kurt


def tail_risk():
    option_data, r, S = getParam()

    data_call, data_put = splitCallPut(option_data)

    data_call, data_put = assetPriceCallPut(data_call, data_put, r)

    call_data, put_data = getOutofMoney(data_call, data_put)

    call_data, put_data = impVolCal(call_data, put_data, r)

    res_df = getRes(call_data, put_data)

    vol_after_kt, tmesh, kmesh = impVolInterp(res_df)

    # 这里可以画一个50etf波动率曲面了，示例代码如下
    # pylab.style.use('ggplot')
    # maturityMesher, strikeMesher = np.meshgrid(tmesh, kmesh)
    # pylab.figure(figsize = (12,7))
    # ax = pylab.gca(projection = '3d')
    # surface1 = ax.plot_surface(strikeMesher, maturityMesher, vol_after_kt*100, cmap = cm.jet)
    # pylab.colorbar(surface1, shrink=0.75)
    # pylab.title("50ETF期权波动率曲面(2017-10-17)")
    # pylab.xlabel('Strike')
    # pylab.ylabel('Maturity')
    # ax.set_zlabel('Volatility(%)')
    # pylab.show()

    call_price = callPricing(vol_after_kt, r, tmesh, kmesh, S)

    # # 这里可以画一个欧式看涨期权定价曲面，示例代码如下
    # pylab.style.use('ggplot')
    # maturityMesher, strikeMesher = np.meshgrid(tmesh, kmesh)
    # pylab.figure(figsize = (12,7))
    # ax = pylab.gca(projection = '3d')
    # surface2 = ax.plot_surface(strikeMesher, maturityMesher, call_price, cmap = cm.jet)
    # pylab.colorbar(surface2, shrink=0.75)
    # pylab.title("欧式看涨期权定价函数(2017-10-17)")
    # pylab.xlabel('Strike')
    # pylab.ylabel('Maturity')
    # ax.set_zlabel('Call Price')
    # pylab.show()

    rx, rp = impProbability(call_price, S, r)

    # # 概率分布图
    # # plot2 = plt.plot(rx, rp, 'b', label='diff')
    # # plt.xlabel('RT')
    # # plt.ylabel('density')
    # # plt.legend(loc=3)  # 设置图示的位置
    # # plt.title('density of RT(2017-10-17)')  # 设置标题
    # # plt.show()  # 显示图片
    #
    #
    ror, probability = gevTail(rx, rp)
    #
    # # 完整概率分布图
    # plt.figure(figsize=(20, 10))
    # plt.title(u'GEV tail')
    # plt.xlabel(u'RT')
    # plt.ylabel(u'density')
    # plt.plot(ror, probability, 'b', label='diff')
    # plt.show()
    #
    imp_vol, imp_skew, imp_kurt = impInfo(probability)
    return tmesh, kmesh, vol_after_kt, call_price, ror, probability, imp_vol, imp_skew, imp_kurt


# tail_risk调用示例
tmesh, kmesh, vol_after_kt, call_price, ror, probability, imp_vol, imp_skew, imp_kurt = tail_risk()