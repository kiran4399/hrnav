# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: server_manager.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='server_manager.proto',
  package='robo_gym_server_manager.grpc_msgs.server_manager',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x14server_manager.proto\x12\x30robo_gym_server_manager.grpc_msgs.server_manager\"T\n\x0bRobotServer\x12\x0b\n\x03\x63md\x18\x01 \x01(\t\x12\x0b\n\x03gui\x18\x02 \x01(\x08\x12\x0c\n\x04port\x18\x03 \x01(\x05\x12\x0f\n\x07success\x18\x04 \x01(\x08\x12\x0c\n\x04info\x18\x05 \x01(\t\"\x07\n\x05\x45mpty\"\x17\n\x06Status\x12\r\n\x05\x61live\x18\x01 \x01(\x08\x32\xce\x04\n\rServerManager\x12\x90\x01\n\x0eStartNewServer\x12=.robo_gym_server_manager.grpc_msgs.server_manager.RobotServer\x1a=.robo_gym_server_manager.grpc_msgs.server_manager.RobotServer\"\x00\x12\x8c\x01\n\nKillServer\x12=.robo_gym_server_manager.grpc_msgs.server_manager.RobotServer\x1a=.robo_gym_server_manager.grpc_msgs.server_manager.RobotServer\"\x00\x12\x90\x01\n\x0eKillAllServers\x12=.robo_gym_server_manager.grpc_msgs.server_manager.RobotServer\x1a=.robo_gym_server_manager.grpc_msgs.server_manager.RobotServer\"\x00\x12\x87\x01\n\x10VerifyConnection\x12\x37.robo_gym_server_manager.grpc_msgs.server_manager.Empty\x1a\x38.robo_gym_server_manager.grpc_msgs.server_manager.Status\"\x00\x62\x06proto3'
)




_ROBOTSERVER = _descriptor.Descriptor(
  name='RobotServer',
  full_name='robo_gym_server_manager.grpc_msgs.server_manager.RobotServer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cmd', full_name='robo_gym_server_manager.grpc_msgs.server_manager.RobotServer.cmd', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gui', full_name='robo_gym_server_manager.grpc_msgs.server_manager.RobotServer.gui', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='port', full_name='robo_gym_server_manager.grpc_msgs.server_manager.RobotServer.port', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='success', full_name='robo_gym_server_manager.grpc_msgs.server_manager.RobotServer.success', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='info', full_name='robo_gym_server_manager.grpc_msgs.server_manager.RobotServer.info', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=74,
  serialized_end=158,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='robo_gym_server_manager.grpc_msgs.server_manager.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=160,
  serialized_end=167,
)


_STATUS = _descriptor.Descriptor(
  name='Status',
  full_name='robo_gym_server_manager.grpc_msgs.server_manager.Status',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='alive', full_name='robo_gym_server_manager.grpc_msgs.server_manager.Status.alive', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=169,
  serialized_end=192,
)

DESCRIPTOR.message_types_by_name['RobotServer'] = _ROBOTSERVER
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['Status'] = _STATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RobotServer = _reflection.GeneratedProtocolMessageType('RobotServer', (_message.Message,), {
  'DESCRIPTOR' : _ROBOTSERVER,
  '__module__' : 'server_manager_pb2'
  # @@protoc_insertion_point(class_scope:robo_gym_server_manager.grpc_msgs.server_manager.RobotServer)
  })
_sym_db.RegisterMessage(RobotServer)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'server_manager_pb2'
  # @@protoc_insertion_point(class_scope:robo_gym_server_manager.grpc_msgs.server_manager.Empty)
  })
_sym_db.RegisterMessage(Empty)

Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'server_manager_pb2'
  # @@protoc_insertion_point(class_scope:robo_gym_server_manager.grpc_msgs.server_manager.Status)
  })
_sym_db.RegisterMessage(Status)



_SERVERMANAGER = _descriptor.ServiceDescriptor(
  name='ServerManager',
  full_name='robo_gym_server_manager.grpc_msgs.server_manager.ServerManager',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=195,
  serialized_end=785,
  methods=[
  _descriptor.MethodDescriptor(
    name='StartNewServer',
    full_name='robo_gym_server_manager.grpc_msgs.server_manager.ServerManager.StartNewServer',
    index=0,
    containing_service=None,
    input_type=_ROBOTSERVER,
    output_type=_ROBOTSERVER,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='KillServer',
    full_name='robo_gym_server_manager.grpc_msgs.server_manager.ServerManager.KillServer',
    index=1,
    containing_service=None,
    input_type=_ROBOTSERVER,
    output_type=_ROBOTSERVER,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='KillAllServers',
    full_name='robo_gym_server_manager.grpc_msgs.server_manager.ServerManager.KillAllServers',
    index=2,
    containing_service=None,
    input_type=_ROBOTSERVER,
    output_type=_ROBOTSERVER,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='VerifyConnection',
    full_name='robo_gym_server_manager.grpc_msgs.server_manager.ServerManager.VerifyConnection',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_STATUS,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_SERVERMANAGER)

DESCRIPTOR.services_by_name['ServerManager'] = _SERVERMANAGER

# @@protoc_insertion_point(module_scope)
