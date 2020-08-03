import torch.nn as nn
import torch
import pickle
import sys
import struct
import fcntl
import warnings
from torch._utils_internal import get_source_lines_and_file
from models import MLP, CNNMnist, CNNCifar

DEFAULT_PROTOCOL = 2
PROTOCOL_VERSION = 1001
LONG_SIZE = struct.Struct('=l').size
INT_SIZE = struct.Struct('=i').size
SHORT_SIZE = struct.Struct('=h').size
MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
_package_registry = []


def register_package(priority, tagger, deserializer):
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()

def _cpu_tag(obj):
    if type(obj).__module__ == 'torch':
        return 'cpu'

def _cpu_deserialize(obj, location):
    if location == 'cpu':
        return obj

register_package(10, _cpu_tag, _cpu_deserialize)

def _is_compressed_file(f):
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False


def _should_read_directly(f):
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False

def location_tag(storage):
    for _, tagger, _ in _package_registry:
        location = tagger(storage)
        if location:
            return location
    raise RuntimeError("don't know how to determine data location of "
                       + torch.typename(storage))

def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
    opened_file = open(f, 'wb')
    fcntl.flock(opened_file.fileno(), fcntl.LOCK_EX)
    _legacy_save(obj, opened_file, pickle_module, pickle_protocol)
    opened_file.close()

def _legacy_save(obj, f, pickle_module, pickle_protocol):
    serialized_container_types = {}
    serialized_storages = {}

    def persistent_id(obj):
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj in serialized_container_types:
                return None
            serialized_container_types[obj] = True
            source_file = source = None
            try:
                source_lines, _, source_file = get_source_lines_and_file(obj)
                source = ''.join(source_lines)
            except Exception:  # saving the source is optional, so we can ignore any errors
                warnings.warn("Couldn't retrieve source code for container of "
                              "type " + obj.__name__ + ". It won't be checked "
                              "for correctness upon loading.")
            return ('module', obj, source_file, source)

        elif torch.is_storage(obj):
            storage_type = normalize_storage_type(type(obj))
            # Offset is always 0, but we keep it for backwards compatibility
            # with the old serialization format (which supported storage views)
            offset = 0
            obj_key = str(obj._cdata)
            location = location_tag(obj)
            serialized_storages[obj_key] = obj
            is_view = obj._cdata != obj._cdata
            if is_view:
                view_metadata = (str(obj._cdata), offset, obj.size())
            else:
                view_metadata = None

            return ('storage',
                    storage_type,
                    obj_key,
                    location,
                    obj.size(),
                    view_metadata)
        return None

    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == 'little',
        type_sizes=dict(
            short=SHORT_SIZE,
            int=INT_SIZE,
            long=LONG_SIZE,
        ),
    )

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)
    pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        serialized_storages[key]._write_file(f, _should_read_directly(f), True)


#global_model = CNNMnist()
#save(global_model,'test.pkl')

#aaa = torch.load('test.pkl')
#print(type(aaa))
