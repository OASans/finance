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

输出为由policy产生的新资产配置list

后置条件：
返回一个Dataframe，目前是只有一列，为总资产的价值，每一行为一个时间戳

## 细节说明：
1. 对于非法asset_id输入,即找不到该id对应股票数据的，目前直接删除该id和对应持仓
2. 目前模拟为以开盘价成交计算
3. 对于策略合法性的自动检验上：
	- 现金不够时按照输入id顺序尽量多地购买；
	- 不满足100股整数倍的，以四舍五入改变持仓
	- 尚未包含手续费计算
4. 尚不支持期权期货（输入没有期权期货的数据；计算上和成交规则上也没有针对期权期货设计），等期权期货数据获取到了我再写吧……
4. 若策略函数需其他输入、回测输出更多其他信息的（如持仓配置的历史变化等）可联系我修改代码 

## 使用样例
```python
# use examples
d=back_test(['000001.SZ','000010.SZ','10001677SH'],[10000,10000,0],100000,policy_stay_calm,'2019-4','2019-8',1)
from matplotlib import pyplot as plt
plt.figure()
plt.plot_date(d.index,d.values,label='1',fmt='-')

dd=back_test(['000001.SZ','000010.SZ','10001677SH'],[10000,10000,0],100000,policy_delta,'2019-4','2019-8',1)
plt.plot_date(dd.index,dd.values,label='2',fmt='-')

plt.legend()
plt.show()
```

可以看出用了delta套保还是勉强有效果的……