import pandas as pd
import threading
import os
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Union, List, Any
from abc import ABC, abstractmethod

import yaml
import copy
from ds_discovery.config.decoratorpatterns import singleton

__author__ = 'Darryl Oatridge'


class AbstractKeys(ABC):
    """ abstract keys class as a super class for referencing keys"""

    def __init__(self, manager: str):
        self._manager = manager
        self.join = PropertyManager.join

    @property
    def items(self) -> iter:
        return self.__dir__()

    @property
    def manager(self) -> str:
        return self._manager


class AbstractPropertiesManager(ABC):
    """ abstract properties class that creates a super class for all properties managers """

    def __init__(self, manager: str):
        """initialises the super properties manager

        :param keys: a concrete instance of an AbstractKeys class
        """
        self._manager = manager
        self._pm = PropertyManager()

    @property
    def config_file_name(self):
        return "{}_{}.{}".format('config', self._manager, 'yaml')

    @abstractmethod
    def KEY(self):
        pass

    def is_key(self, key: str) -> bool:
        """test if the key exists"""
        return self._pm.is_key(key)

    def get(self, key: str) -> Union[object, str, dict, tuple, list]:
        """gets a property value for the dot separated key"""
        return self._pm.get(key)

    def set(self, key: str, value: Any):
        """ sets the value for the dot separated key to the properties manager"""
        return self._pm.set(key, value)

    def remove(self, key: str):
        """ removes the dot separated key from the properties manager"""
        return self._pm.remove(key)

    def join(self, *names, sep=None) -> str:
        """Used to create a name string. Can also be used to join paths by passing sep=os.path.sep"""
        return self._pm.join(*names, sep=sep)

    def save(self, config_file:str=None, use_base_key:bool=True):
        """Saves the properties to a configuration file"""
        if config_file is None or len(config_file) == 0:
            config_file = self.config_file_name
        _key = self._manager if use_base_key else None
        self._pm.dump(config_file, key=_key)

    def _get_all(self) -> dict:
        return self._pm.get_all()

    def load(self, config_file=None, replace=False) -> bool:
        if config_file is None or len(config_file) == 0:
            config_file = self.config_file_name
        return self._pm.load(config_file=config_file, replace=replace)

    @staticmethod
    def list_formatter(value) -> [List[str], list, None]:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series into a list"""
        if isinstance(value, (int, float, str, pd.Timestamp)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return  list(value)
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, dict):
            return list(value.items())
        return None


class PropertyManager(object):
    """

    A thread safe singleton configuration class that allows for the management of key/value pairs
    to be stored and retrieved. The persisted values are stored in YAML files specified when persisted

    The value's are stored as a tree structure with the key being a dot separated string value up the tree.
    For example key 'root.directories.base_dir' returns [str]: 'filepath' .
    Where the underlying Dictionary looks like { root: { directories: { base_dir : 'filepath' }}}.

    The key can reference any part of the tree and will return the object at that point.
    From the example above key 'root' would return [dict]: { directories: { base_dir : 'filepath' }}.

    The key must start from the base key and work up, this allows namespace to avoid repeased key values.
    """

    __properties = dict({})
    _persisted_path = None
    _persisted_names = list([])

    @singleton
    def __new__(cls):
        return super().__new__(cls)

    @classmethod
    def is_key(cls, key) -> bool:
        """identifies if a key exists or not.

        :param key: the key of the value
            The key should be a dot separated string of keys from root up the tree
        :return:
            True if the key exists in the properties
            False if the key doesn't exist in the properties
        """
        if key is None or not key:
            return False
        find_dict = cls.__properties
        is_path, _, is_key = key.rpartition('.')
        if is_path:
            for part in is_path.split('.'):
                if isinstance(find_dict, dict):
                    find_dict = find_dict.get(part, {})
                else:
                    break
        if is_key in find_dict:
            return True
        return False

    @classmethod
    def get(cls, key) -> Union[object, str, dict, tuple, list]:
        """ gets a property value for the dot separated key.

        :param key: the key of the value
            The key should be a dot separated string of keys from root up the tree

        :return:
            an object found in the key can be any structure found under that key
            if the key is not found, None is returned
            If the key is None then the complete properties dictionary is returned
            will be the full tree under the requested key, be it a value, tuple, list or dictionary
        """
        if key is None or not key:
            return None
        rtn_val = cls.__properties
        for part in key.split('.'):
            if isinstance(rtn_val, dict):
                rtn_val = rtn_val.get(part)
                if rtn_val is None:
                    return None
            else:
                return None
        with threading.Lock():
            return copy.deepcopy(rtn_val)

    @classmethod
    def get_all(cls) -> dict:
        """ gets all the properties

        :returns:
            a deep copy of the  of key/value pairs
        """
        with threading.Lock():
            return copy.deepcopy(cls.__properties)

    @classmethod
    def set(cls, key, value) -> None:
        """ sets a key value pair. The value acan be Union(Str, Dict, Tuple, Array)

        :param key: the key string
        :param value: the value of the key
        """
        if key is None or not key or not isinstance(key, str):
            raise ValueError("The key must be a valid str")
        keys = key.split('.')
        _prop_branch = cls.__properties
        _last_key = None
        _last_prop_branch = None
        # from base of the key work up to find where the section doesn't exist
        for _, k in list(enumerate(keys, start=0)):
            if k not in _prop_branch:
                break
            _last_prop_branch = _prop_branch
            _last_key = k
            _prop_branch = _prop_branch[k]
        tmp_dict = {}
        # now from the top of the key work back, creating the sections tree
        k = None
        for _, k in reversed(list(enumerate(keys, start=0))):
            if isinstance(value, dict):
                tmp_dict = {k: value}
            else:
                tmp_dict[k] = value
            if k is _last_key:
                break
            value = tmp_dict
        if not isinstance(value, dict):
            if isinstance(_last_prop_branch[k], list):
                if isinstance(value, list):
                    _last_prop_branch[k] += value
                else:
                    _last_prop_branch[k].append(value)
            else:
                _last_prop_branch[k] = value
            return
        if _last_prop_branch is None:
            _prop_branch.update(value)
        else:
            cls._add_value(k, value, _last_prop_branch)
        return

    @classmethod
    def remove(cls, key) -> bool:
        """removes a key/value from the in-memory configuration dictionary based on the key

        :param key: the key of the key/value to be removed
            The key should be a dot separated string of keys from root up the tree

        :return:
            True if the key was removed
            False if the key was not found
        """
        del_dict = cls.__properties
        del_path, _, del_key = key.rpartition('.')
        if del_path:
            for part in del_path.split('.'):
                if isinstance(del_dict, dict):
                    del_dict = del_dict.get(part)
                else:
                    return False
        if del_dict is None or del_key not in del_dict:
            return False
        with threading.Lock():
            _ = del_dict.pop(del_key, None)
        return True

    @classmethod
    def get_property_store(cls) -> (str, list):
        return cls._persisted_path, cls._persisted_names

    @classmethod
    def set_property_store(cls, path, name):
        cls._persisted_path = path
        cls._persisted_names.append(name)

    @classmethod
    def has_property_path(cls):
        if cls._persisted_path is None or not cls._persisted_path:
            return False
        if not os.path.exists(cls._persisted_path):
            return False
        return True

    @classmethod
    def load(cls, config_file, replace=False) -> bool:
        """ loads the properties from the yaml configuration file. allows for multiple configuration
        files to be merged into the properties dictionary, or properties to be refreshed in real time.

        :param config_file: The path and filename of the YAML file.
        :param replace: if the file is to be added to or replaced
        :return:
            True if the file was found, opened and loaded, False if an exception was thrown.

        """
        if config_file is None or not config_file or not isinstance(config_file, str):
            raise ValueError("The properties configuration file must be a valid str")

        _path = Path(config_file)
        if not _path.exists() or not _path.is_file():
            raise FileNotFoundError("Can't find the configuration file {}".format(_path))

        try:
            cfg_dict = cls._yaml_load(_path)
        except (IOError, TypeError) as e:
            raise IOError(e)
        if replace:
            with threading.Lock():
                cls.__properties.clear()
                cls.__properties = cfg_dict
            return True
        for _key, _value in cfg_dict.items():
            # only replace sections that have changed
            if cls.is_key(_key) and cfg_dict.get(_key) == cls.get(_key):
                continue
            cls.remove(_key)
            cls.set(_key, _value)
        return True

    @classmethod
    def dump(cls, config_file, key=None) -> None:
        """ Dumps the current in-memory configuration to a persistence store.
        Note that this replaces existing content if the file exists. This is particularly
        important is only persisting a single root key.

        The use of the root key option allows for the breakdown of persisted files into
        multiple files for easier management.

        :param config_file: the name of the file to dump to.
        :param key: An optional root key subset of the configuration values.
        """
        # set the config file meta data
        if config_file is None or not config_file or not isinstance(config_file, str):
            raise ValueError("The properties configuration file must be a valid str")
        _time = str(datetime.now())
        if not cls.is_key('config_meta.uid') or not cls.is_key('config_meta.create'):
            cls.set('config_meta.uid', int(round(time.time() * 1000)))
            cls.set('config_meta.create', _time)
        cls.set('config_meta.modify', _time)
        # check we want to persist
        data = {}
        if key is None:
            data = cls.__properties
        else:
            if key in cls.__properties.keys():
                data[key] = cls.get(key)
                data['config_meta'] = cls.get('config_meta')
            else:
                raise KeyError("The key {} must be at the root of the properties".format(key))

        if config_file is None:
            raise ValueError("The config_file can't be value None")

        _path = Path(config_file)
        # now add the data
        cls._yaml_dump(data, _path)
        return

    @staticmethod
    def join(*names, sep=None) -> str:
        """Used to create a name string. Can also be used to join paths by passing sep=os.path.sep

        :param names: the names to join
        :param sep: the join separator. Default to '.'
        :return: the names joined with the separator
        """
        _sep = sep if sep is not None else '.'
        return _sep.join(map(str, names))

    @classmethod
    def _remove_all(cls):
        with threading.Lock():
            cls.__properties.clear()

    @classmethod
    def _add_value(cls, key, value, root):
        if key is None:
            return
        for k, v in value.items():
            if isinstance(v, dict) and k in root[key]:
                cls._add_value(k, v, root[key])
            else:
                with threading.Lock():
                    if k in root[key] and isinstance(root[key][k], dict):
                        root[key][k].update(v)
                    else:
                        if k not in root[key] and not isinstance(root[key], dict):
                            root[key] = {}
                        root[key][k] = v
        return

    @staticmethod
    def _yaml_dump(data, config_file) -> None:
        _path, _file = os.path.split(config_file)
        if _path is not None and len(_path) > 0 and isinstance(_path, str) and not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
        _config_file = Path(config_file)
        with threading.Lock():
            # make sure the dump is clean
            try:
                with closing(open(_config_file, 'w')) as ymlfile:
                    yaml.safe_dump(data, ymlfile, default_flow_style=False)
            except IOError as e:
                raise IOError("The configuration file {} failed to open with: {}".format(config_file, e))
        # check the file was created
        if not _config_file.exists():
            raise IOError("Failed to save configconfiguration file {}. Check the disk is writable".format(config_file))
        return

    @staticmethod
    def _yaml_load(config_file) -> dict:
        _config_file = Path(config_file)
        if not _config_file.exists():
            raise FileNotFoundError("The configuration file {} does not exist".format(config_file))
        with threading.Lock():
            try:
                with closing(open(_config_file, 'r')) as ymlfile:
                    rtn_dict = yaml.safe_load(ymlfile)
            except IOError as e:
                raise IOError("The configuration file {} failed to open with: {}".format(config_file, e))
            if not isinstance(rtn_dict, dict) or not rtn_dict:
                raise TypeError("The configuration file {} could not be loaded as a dict type".format(config_file))
            return rtn_dict
