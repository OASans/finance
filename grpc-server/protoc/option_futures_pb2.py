# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: option_futures.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='option_futures.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x14option_futures.proto2\x0f\n\rOptionFuturesb\x06proto3')
)



_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_OPTIONFUTURES = _descriptor.ServiceDescriptor(
  name='OptionFutures',
  full_name='OptionFutures',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=24,
  serialized_end=39,
  methods=[
])
_sym_db.RegisterServiceDescriptor(_OPTIONFUTURES)

DESCRIPTOR.services_by_name['OptionFutures'] = _OPTIONFUTURES

# @@protoc_insertion_point(module_scope)
