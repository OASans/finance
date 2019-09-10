## 回测函数部分

语法：

`back_test(begin_asset_id:list<str>,begin_asset_amount:list<int>,begin_cash:float,policy:function,begin_t:str,end_t:str,delta_t:int)`

描述:根据给定初始资产配置、给定策略函数进行回测模拟

前置条件：

begin_asset_id 为id的list ,如['000001.SZ','000002.SZ']
begin_asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
begin_cash为初始资金数量的浮点数
begin_t、end_t 为 str类型时间戳，如'2019-8-1'、'2019-08-01'、'2019-8'均合法
delta_t为整型为触发回测函数的天数，直接取1吧
policy 为策略函数，计划实现4种：policy_stay_calm、policy_delta、policy_gamma、policy_beta，分别对应默认什么也不做的策略和3个套保页面的策略，两者（policy_stay_calm和其他一种）画在同一张图中做对比

具体策略的要求：

- policy_delta要求begin_asset_id最后一个是选用的套保期权
- policy_gamma要求begin_asset_id最后两个是选用的套保期权
- policy_beta要求begin_asset_id最后一个是选用的套保期货

输出为由policy产生的新资产配置list

后置条件：
返回一个Dataframe，目前是只有一列，为总资产的价值，每一行为一个时间戳

## 使用样例
```python
# use examples
d=back_test(['000001.SZ','10001689.SH','10001681.SH'],[100000,0,0],100000,policy_stay_calm,'2019-1','2019-9',1)
from matplotlib import pyplot as plt
plt.figure()
plt.plot_date(d.index,d.values,label='No Hedging',fmt='-')

dd=back_test(['000001.SZ','10001689.SH','10001681.SH'],[100000,0,0],100000,policy_delta,'2019-1','2019-9',1)#10001677SH

plt.plot_date(dd.index,dd.values,label='ML-Delta Dynamic Hedging',fmt='-')

ddd=back_test(['000001.SZ','10001689.SH','10001681.SH'],[100000,0,0],100000,policy_gamma,'2019-1','2019-9',1)
plt.plot_date(ddd.index,ddd.values,label='Beta Hedging',fmt='-')
plt.legend()
plt.show()
```

可以看出用了delta套保还是勉强有效果的……