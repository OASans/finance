def poundage(DealAmount, BuyIn=True, BCRate=0.002):
    
    # 用户仅在卖出股票时要缴纳印花税
    if BuyIn:
        StampDuty = 0
    else:
        StampDuty = DealAmount * 0.001

    # 券商交易佣金最高不超过成交金额的0.3%，单笔交易佣金不满5元按5元收取
    BrokerageCommission = max(DealAmount * min(0.003, BCRate), 5)

    # 过户费
    TransferFee = DealAmount * 0.002

    # 总股票交易手续费
    Poundage = StampDuty + BrokerageCommission + TransferFee

    # 返回结果
    return Poundage
