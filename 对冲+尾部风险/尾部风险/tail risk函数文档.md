# 尾部风险部分

## 语法：

`tail_risk(option_data:DataFrame, r:float, S:float)`

### 描述: 基于当日的50etf期权情况，计算隐含概率分布与尾部风险相关指标

### 前置条件：

* **option_data**: 

  当日可获取的所有在交易的50etf期权信息, DataFrame格式

  必要信息如图![image-20190829005510490](/Users/ishi/Library/Application Support/typora-user-images/image-20190829005510490.png)

* **r**：无风险利率，取SHIBOR（3M）
* **S**：标的资产价格

### 后置条件：

- vol_after_kt: 画波动率曲面
- call_price: 画看涨期权定价曲面

* tmesh，kmesh: 画vol_after_kt和call_price图用的，是插值后的坐标
* ror, probability：期权隐含概率分布图的x、y轴
* imp_vol: 市场隐含波动率
* imp_skew: 市场隐含偏度
* imp_kurt：市场隐含峰度



## 细节说明：

1. **option_data**: 参考**windApi** `w.wset("optionchain","date=2017-10-17; us_code=510050.SH;option_var=全部;call_put=全部")`的返回值所包含的内容
   windApi获取的返回值转换为DataFrame后为下图（供参考）![Screenshot 2019-08-29 at 00.42.53](./图示意/Screenshot 2019-08-29 at 00.42.53.png)必须包含的信息有'option_code', 'exe_type', 'strike_price', 'call_put', 'expiredate'（ '期权代码', '期权类型', '行权价格', '看涨/看跌', '剩余存续期'）

   以及根据option_code, 从windApi获取（参考）的期权收盘价close`w.wss(list(option_data['option_code']), 'close', 'tradeDate=2017-10-17').Data[0]`

   二者拼起来就是上面图展示的DataFrame

2. **S**：参考windApi`w.wsd("510050.SH", "close", "2017-10-17", "2017-10-17", "Fill=Previous", usedf=True)`

3. 由于期权数据现在还不知道是否能获取、如何获取、系统内存储方式、包含信息及格式等，目测这代码没法直接用。强烈建议数据情况更新后联系我修改

   

## 画图示例
![Screenshot 2019-08-27 at 15.49.53](./图示意/Screenshot 2019-08-27 at 15.49.53.png)

![Screenshot 2019-08-27 at 15.50.05](./图示意/Screenshot 2019-08-27 at 15.50.05.png)

![Screenshot 2019-08-27 at 15.50.33](./图示意/Screenshot 2019-08-27 at 15.50.33.png)

![Screenshot 2019-08-27 at 15.50.43](./图示意/Screenshot 2019-08-27 at 15.50.43.png)