# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import adjustment_and_triggering_of_portfolio_pb2 as adjustment__and__triggering__of__portfolio__pb2


class AdjustmentAndTriggeringOfPortfolioStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.PortFolioVar = channel.unary_unary(
        '/AdjustmentAndTriggeringOfPortfolio/PortFolioVar',
        request_serializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioInput.SerializeToString,
        response_deserializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioOutput.FromString,
        )
    self.PortfolioVolatility = channel.unary_unary(
        '/AdjustmentAndTriggeringOfPortfolio/PortfolioVolatility',
        request_serializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioInput.SerializeToString,
        response_deserializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioOutput.FromString,
        )
    self.PortfolioDiff = channel.unary_unary(
        '/AdjustmentAndTriggeringOfPortfolio/PortfolioDiff',
        request_serializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioInput.SerializeToString,
        response_deserializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioOutput.FromString,
        )


class AdjustmentAndTriggeringOfPortfolioServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def PortFolioVar(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def PortfolioVolatility(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def PortfolioDiff(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_AdjustmentAndTriggeringOfPortfolioServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'PortFolioVar': grpc.unary_unary_rpc_method_handler(
          servicer.PortFolioVar,
          request_deserializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioInput.FromString,
          response_serializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioOutput.SerializeToString,
      ),
      'PortfolioVolatility': grpc.unary_unary_rpc_method_handler(
          servicer.PortfolioVolatility,
          request_deserializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioInput.FromString,
          response_serializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioOutput.SerializeToString,
      ),
      'PortfolioDiff': grpc.unary_unary_rpc_method_handler(
          servicer.PortfolioDiff,
          request_deserializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioInput.FromString,
          response_serializer=adjustment__and__triggering__of__portfolio__pb2.PortFolioOutput.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'AdjustmentAndTriggeringOfPortfolio', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
