def stock_poundage(DealAmount, BuyIn=True, BCRate=0.003):

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
    StockPoundage = StampDuty + BrokerageCommission + TransferFee

    # 返回结果
    return StockPoundage


def future_poundage(DealAmount): # 成交金额

    # 手续费主要部分
    MainPoundage = DealAmount * 0.000025

    # 保障基金
    Fund = DealAmount * 0.00002

    # 总期货交易手续费
    FuturePoundage = MainPoundage + Fund

    return FuturePoundage


def options_poundage(NumOfPiece, BCUnitPrice=5): # 成交张数

    # 交易经手费
    Fee_1 = NumOfPiece * 1.3

    # 交易结算费
    Fee_2 = NumOfPiece * 0.3

    # 佣金
    TransferFee = NumOfPiece * max(BCUnitPrice, 2)

    #总期权交易手续费
    OptionsPoundage = Fee_1 + Fee_2 + TransferFee

    return OptionsPoundage