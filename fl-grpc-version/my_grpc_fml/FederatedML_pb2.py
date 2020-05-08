# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: FederatedML.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='FederatedML.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x11\x46\x65\x64\x65ratedML.proto\".\n\x05Model\x12\r\n\x05model\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\x05\x12\n\n\x02gr\x18\x03 \x01(\x05\" \n\x06Option\x12\n\n\x02op\x18\x01 \x01(\x05\x12\n\n\x02gr\x18\x02 \x01(\x05\"\x16\n\x05\x45mpty\x12\r\n\x05value\x18\x01 \x01(\x05\x32K\n\x0b\x46\x65\x64\x65ratedML\x12\x1d\n\x08GetModel\x12\x07.Option\x1a\x06.Model\"\x00\x12\x1d\n\tSendModel\x12\x06.Model\x1a\x06.Empty\"\x00\x62\x06proto3'
)




_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='Model.model', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='Model.id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gr', full_name='Model.gr', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=21,
  serialized_end=67,
)


_OPTION = _descriptor.Descriptor(
  name='Option',
  full_name='Option',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='op', full_name='Option.op', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gr', full_name='Option.gr', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=69,
  serialized_end=101,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Empty.value', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=103,
  serialized_end=125,
)

DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['Option'] = _OPTION
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
  'DESCRIPTOR' : _MODEL,
  '__module__' : 'FederatedML_pb2'
  # @@protoc_insertion_point(class_scope:Model)
  })
_sym_db.RegisterMessage(Model)

Option = _reflection.GeneratedProtocolMessageType('Option', (_message.Message,), {
  'DESCRIPTOR' : _OPTION,
  '__module__' : 'FederatedML_pb2'
  # @@protoc_insertion_point(class_scope:Option)
  })
_sym_db.RegisterMessage(Option)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'FederatedML_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  })
_sym_db.RegisterMessage(Empty)



_FEDERATEDML = _descriptor.ServiceDescriptor(
  name='FederatedML',
  full_name='FederatedML',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=127,
  serialized_end=202,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetModel',
    full_name='FederatedML.GetModel',
    index=0,
    containing_service=None,
    input_type=_OPTION,
    output_type=_MODEL,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendModel',
    full_name='FederatedML.SendModel',
    index=1,
    containing_service=None,
    input_type=_MODEL,
    output_type=_EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_FEDERATEDML)

DESCRIPTOR.services_by_name['FederatedML'] = _FEDERATEDML

# @@protoc_insertion_point(module_scope)
