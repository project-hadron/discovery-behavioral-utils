import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import string
import pandas as pd
from pandas.tseries.offsets import Week
import matplotlib.dates as mdates

from ds_discovery.config.properties import AbstractPropertiesManager
from ds_behavioral.sample.sample_data import ProfileSample
from ds_discovery.transition.cleaners import ColumnCleaners as Cleaner

__author__ = 'Darryl Oatridge'


class DataBuilderPropertyManager(AbstractPropertiesManager):

    def __init__(self, build_name: str):
        if build_name is None or not build_name or not isinstance(build_name, str):
            assert TypeError("The build_name can't be None or of zero length. '{}' passed".format(build_name))
        keys = ['columns', 'correlate', 'associate']
        super().__init__(manager='data_builder', contract=build_name, keys=keys)
        if not self.is_key(self.KEY.contract_key):
            self.set(self.KEY.contract_key, {})

    @property
    def KEY(self):
        return self._keys

    @property
    def builder(self):
        """returns true if the key exists"""
        if self.is_key(self.KEY.contract_key):
            return self.get(self.KEY.contract_key)
        return {}

    @property
    def columns(self):
        """returns a list of columns"""
        rtn_col = []
        if self.is_key(self.KEY.columns_key):
            rtn_col.extend(self.get(self.KEY.columns_key).keys())
        if self.is_key(self.KEY.correlate_key):
            rtn_col.extend(self.get(self.KEY.correlate_key).keys())
        return rtn_col

    def get_column(self, name: str):
        _column_key = self.join(self.KEY.columns_key, name)
        _association_key = self.join(self.KEY.correlate_key, name)
        for _key in [_column_key, _association_key]:
            if self.is_key(_key):
                return self.get(_key)
        return {}

    def set_column(self, label: str, etype: str, **kwargs):
        _key = self.join(self.KEY.columns_key, label)
        self.set(self.join(_key, 'etype'), etype)
        self.set(self.join(_key, 'kwargs'), {})
        for k, v in kwargs.items():
            self.set(self.join(_key, 'kwargs', k), v)

    def set_association(self, label: str, associate: [str, list], etype: str, **kwargs):
        _key = self.join(self.KEY.correlate_key, label)
        associate = self.list_formatter(associate)
        self.set(self.join(_key, 'associate'), associate)
        self.set(self.join(_key, 'etype'), etype)
        self.set(self.join(_key, 'kwargs'), {})
        for k, v in kwargs.items():
            self.set(self.join(_key, 'kwargs', k), v)


class DataBuilder(object):

    def __init__(self, build_name: str):
        """ creates a DataBuilder instance with the reference name given.
        The build_name allows each FileBuild instance to have their wn configuration build.

        :param build_name: a build reference name
        """
        if build_name is None or not build_name or not isinstance(build_name, str):
            assert TypeError("The file_name can't be None or of zero length. '{}' passed".format(build_name))
        self._file_builder_pm = DataBuilderPropertyManager(build_name)
        self._tools = DataBuilderTools()

    @property
    def fbpm(self) -> DataBuilderPropertyManager:
        """
        :return: the file builder properties instance
        """
        return self._file_builder_pm

    @property
    def tools(self):
        """
        :return: the file builder tools instance
        """
        return self._tools

    @property
    def tool_dir(self):
        return self._tools.__dir__()

    def add_column(self, label: str, etype: str, **kwargs) -> None:
        """ Add a column to the build configuration

        :param label: the label or header for the column
        :param etype: the DataBuilderTools method name to execute
        :param kwargs: the key/word args to pass to the method.
        """
        self.fbpm.set_column(label=label, etype=etype, **kwargs)

    def add_association(self, label: str, associate: str, etype: str, **kwargs) -> None:
        """ adds an association column to the configuration

        :param label: the label or header for the column
        :param associate:
        :param etype: the DataBuilderTools method name to execute
        :param kwargs:
        :return:
        """
        self.fbpm.set_association(label=label, associate=associate, etype=etype, **kwargs)

    def build_columns(self, rows: int, filename: str=None) -> pd.DataFrame:
        """ build a file based on the columns tha have been added by column

        :param rows: the number of rows to create
        :param filename: (optional) the filename to output the results to
        :return: pandas Dataframe
        """
        df = pd.DataFrame(index=list(range(rows)))
        for col in self.fbpm.columns:
            col_dict = self.fbpm.get_column(col)
            method_str = "DataBuilderTools.{}(**_kwargs)".format(col_dict['etype'])
            _kwargs = {}
            if 'kwargs' in col_dict.keys():
                _kwargs = col_dict['kwargs']
            _kwargs['size'] = rows
            df.insert(loc=0, column=col, value=eval(method_str))
        if filename is not None:
            df.to_csv(filename, sep=',', index=False)
        return df

    @staticmethod
    def save_to_disk(df, filename: str, path: str=None, file_format: str=None):
        """ Saves a dataframe to disk

        :param df: the dataframe to save
        :param filename: the filename. Note, include the extension
        :param path: (optional) an existing path
        :param file_format: (optional) the format to save too, currently only 'csv' and 'pickle' supported
        """
        _filename = filename
        if path is not None:
            if os.path.exists(path):
                _filename = os.path.join(path, filename)
            else:
                raise FileNotFoundError("The path {} does not exist".format(path))
        if file_format == 'pickle':
            pd.to_pickle(df, path=_filename)
        else:
            df.to_csv(_filename, index=False)


class DataBuilderTools(object):

    def __dir__(self):
        rtn_list = []
        for m in dir(DataBuilderTools):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def get_custom(code_str: str, quantity: float=None, size: int=None, seed: int=None, **kwargs):
        """returns a number based on the random func. The code should generate a value per line
        example:
            code_str = 'round(np.random.normal(loc=loc, scale=scale), 3)'
            fbt.get_custom(code_str, loc=0.4, scale=0.1)

        :param code_str: an evaluable code as a string
        :param quantity: (optional) a number between 0 and 1 representing data that isn't null
        :param size: (optional) the size of the sample
        :param seed: (optional) a seed value for the random function: default to None
        :return: a random value based on function called
        """
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        _seed = DataBuilderTools._seed() if seed is None else seed

        rtn_list = []
        for _ in range(size):
            _seed = DataBuilderTools._next_seed(_seed, seed)
            local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
            rtn_list.append(eval(code_str, globals(), local_kwargs))
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def get_distribution(method: str=None, offset: float=None, precision: int=None, size: int=None,
                         quantity: float=None, seed: int=None, **kwargs):
        """returns a number based the distribution type. Supports Normal, Beta and

        :param method: any method name under np.random. Default is 'normal'
        :param offset: a value to offset the number by. n * offset
        :param precision: the precision of the returned number
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param kwargs: the parameters of the method
        :return: a random number
        """
        offset = 1 if offset is None or not isinstance(offset, (float, int)) else offset
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        _seed = DataBuilderTools._seed() if seed is None else seed

        method = 'normal' if method is None else method
        precision = 3 if precision is None else precision
        func = "np.random.{}(**{})".format(method, kwargs)
        rtn_list = []
        for _ in range(size):
            _seed = DataBuilderTools._next_seed(_seed, seed)
            rtn_list.append(round(eval(func) * offset, precision))
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def get_intervals(intervals: list, weight_pattern: list=None, precision: int=None, currency: str=None,
                      size: int=None, quantity: float=None, seed: int=None):
        """ returns a number based on a list selection of tuple(lower,upper) interval

        :param intervals: a list of unique tuple pairs representing the inteval lower and upper boundaries
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param currency: a currency symbol to prefix the value with. returns string with commas
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        _seed = DataBuilderTools._seed() if seed is None else seed
        interval_list = DataBuilderTools.get_category(selection=intervals, weight_pattern=weight_pattern, size=size,
                                                      seed=_seed)
        interval_counts = pd.Series(interval_list).value_counts()
        rtn_list = []
        for index in interval_counts.index:
            size = interval_counts[index]
            if size == 0:
                continue
            (lower, upper) = index
            rtn_list = rtn_list + DataBuilderTools.get_number(lower, upper, precision=precision, currency=currency,
                                                              size=size, seed=_seed)
        np.random.seed(_seed)
        np.random.shuffle(rtn_list)
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def get_number(from_value: [int, float], to_value: [int, float]=None, weight_pattern: list=None,
                   precision: int=None, currency: str=None, size: int=None, quantity: float=None, seed: int=None):
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param currency: a currency symbol to prefix the value with. returns string with commas
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        (from_value, to_value) = (0, from_value) if not isinstance(to_value, (float, int)) else (from_value, to_value)
        _seed = DataBuilderTools._seed() if seed is None else seed
        is_int = True if isinstance(to_value, int) and isinstance(from_value, int) else False
        precision = 3 if not isinstance(precision, int) else precision
        value_bins = None
        if from_value == to_value:
            weight_pattern = None
        if weight_pattern is not None:
            value_bins = pd.interval_range(start=from_value, end=to_value, periods=len(weight_pattern),  closed='both')
            value_bins.drop_duplicates()
        rtn_list = []
        for _ in range(size):
            _seed = DataBuilderTools._next_seed(_seed, seed)
            if weight_pattern is not None:
                pattern = DataBuilderTools._normailse_weights(weight_pattern, size=size, count=len(rtn_list),
                                                              length=value_bins.size)
                index = DataBuilderTools._weighted_choice(pattern, seed=_seed)
                from_value = value_bins[index].left
                to_value = value_bins[index].right
            value = np.round(np.random.uniform(low=from_value, high=to_value), precision)
            if is_int:
                value = int(value)
            if isinstance(currency, str):
                value = '{}{:0,.{}f}'.format(currency, value, precision)
            rtn_list.append(value)
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def get_reference(header: str, filename: str, weight_pattern: list = None, selection_size: int=None,
                      sample_size: int=None, size: int=None, at_most: bool=None, shuffled: bool=True,
                      file_format: str=None, quantity: float=None, seed: int=None, **kwargs):
        """
        :param header: the name of the header to be selected from
        :param filename: the full path and filename to be loaded
        :param weight_pattern: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, norally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param at_most: (optional) the most times a selection should be chosen
        :param shuffled: (optional) if the selection should be shuffled before selection. Default is true
        :param file_format: (optional) the format of the file to reference
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param size: (optional) size of the return. default to 1
        :param seed: (optional) a seed value for the random function: default to None
        :param kwargs: (optional) key word arguments to pass to the pandas read method
        :return:
        """
        quantity = DataBuilderTools._quantity(quantity)
        _seed = DataBuilderTools._seed() if seed is None else seed
        if not isinstance(file_format, str) and file_format not in ['csv', 'pickle']:
            file_format = 'csv'
        _path = Path(filename)
        if not _path.exists():
            raise ValueError("The filename '{}' does not exist".format(filename))
        if file_format == 'pickle':
            df = pd.read_pickle(_path, **kwargs)
            df = df.iloc[:sample_size]
        else:
            df = pd.read_csv(_path, nrows=sample_size, **kwargs)
        if df.shape[0] == 0:
            raise LookupError("No data was loaded from the file filename '{}'". format(filename))
        if header not in df.columns:
            raise ValueError("The header '{}' does not exist in the ".format(header))
        rtn_list = df[header].tolist()
        if shuffled:
            np.random.seed(_seed)
            np.random.shuffle(rtn_list)
        if isinstance(selection_size, int) and 0 < selection_size < len(rtn_list):
            rtn_list = rtn_list[:selection_size]
        return DataBuilderTools.get_category(selection=rtn_list, weight_pattern=weight_pattern,
                                             quantity=quantity, size=size, at_most=at_most, seed=_seed)

    @staticmethod
    def get_category(selection: list, weight_pattern: list=None, quantity: float=None, size: int=None,
                     at_most: int=None, seed: int=None):
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chocen.

        :param selection: a list of items to select from
        :param weight_pattern: a weighting pattern that does not have to add to 1
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: an optional size of the return. default to 1
        :param at_most: the most times a selection should be chocen
        :param seed: a seed value for the random function: default to None
        :return: an item or list of items chosen from the list
        """
        if not isinstance(selection, list) or len(selection) == 0:
            return [None]*size
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        at_most = None if not isinstance(at_most, int) else at_most
        _seed = DataBuilderTools._seed() if seed is None else seed
        weight_pattern = [1] if not isinstance(weight_pattern, list) else weight_pattern
        if at_most is not None and at_most * len(selection) < size:
            raise ValueError("the selection size '{}' is smaller than the required sample size '{}'".format(
                at_most * len(selection), size))
        selection = selection.copy()
        rtn_list = []
        at_most_counter = [0] * len(selection)
        for _ in range(size):
            if len(selection) == 1:
                rtn_list.append(selection[0])
                continue
            _seed = DataBuilderTools._next_seed(_seed, seed)
            pattern = DataBuilderTools._normailse_weights(weight_pattern, size=size, count=len(rtn_list),
                                                          length=len(selection))
            choice = selection[DataBuilderTools._weighted_choice(pattern, seed=_seed)]
            rtn_list.append(choice)
            if at_most is not None:
                choice_idx = selection.index(choice)
                at_most_counter[choice_idx] += 1
                if at_most_counter[choice_idx] >= at_most:
                    selection.remove(choice)
                    at_most_counter.pop(choice_idx)
        return list(DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed))

    @staticmethod
    def get_string_pattern(pattern: str, choices: dict=None, quantity: [float, int]=None, size: int=None,
                           choice_only: bool=None, seed: int=None):
        """ Returns a random string based on the pattern given. The pattern is made up from the choices passed but
            by default is as follows:
                c = random char [a-z][A-Z]
                d = digit [0-9]
                l = lower case char [a-z]
                U = upper case char [A-Z]
                p = all punctuation
                s = space
            you can also use punctuation in the pattern that will be retained
            A pattern example might be
                    uuddsduu => BA12 2NE or dl-{uu} => 4g-{FY}

            to create your own choices pass a dictionary with a reference char key with a list of choices as a value

        :param pattern: the pattern to create the string from
        :param choices: an optional dictionary of list of choices to replace the default.
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: the size of the return list. if None returns a single value
        :param choice_only: if to only use the choices given or to take not found characters as is
        :param seed: a seed value for the random function: default to None
        :return: amstring based on the pattern
        """
        choice_only = True if choice_only is None or not isinstance(choice_only, bool) else choice_only
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        _seed = DataBuilderTools._seed() if seed is None else seed
        if choices is None or not isinstance(choices, dict):
            choices = {'c': list(string.ascii_letters),
                       'd': list(string.digits),
                       'l': list(string.ascii_lowercase),
                       'U': list(string.ascii_uppercase),
                       'p': list(string.punctuation),
                       's': [' '],
                       }
            choices.update({p: [p] for p in list(string.punctuation)})
        else:
            for k, v in choices.items():
                if not isinstance(v, list):
                    raise ValueError(
                        "The key {} must contain a list of replacements opotions. {} type found".format(k, type(v)))

        rtn_list = []
        for _ in range(size):
            _seed = DataBuilderTools._next_seed(_seed, seed)
            result = []
            for c in list(pattern):
                if c in choices.keys():
                    result.append(np.random.choice(choices[c]))
                elif not choice_only:
                    result.append(c)
            rtn_list.append(''.join(result))
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def get_datetime(start: Any, until: Any, default: Any = None, ordered: bool=None,
                     date_pattern: list = None, year_pattern: list = None, month_pattern: list = None,
                     weekday_pattern: list = None, hour_pattern: list = None, minute_pattern: list = None,
                     quantity: float = None, date_format: str = None, size: int = None, seed: int = None,
                     day_first: bool = True, year_first: bool = False):
        """ returns a random date between two date and times. weighted patterns can be applied to the overall date
        range, the year, month, day-of-week, hours and minutes to create a fully customised random set of dates.
        Note: If no patterns are set this will return a linearly random number between the range boundaries.
              Also if no patterns are set and a default date is given, that default date will be returnd each time

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param default: (optional) a fixed starting date that patterns are applied too.
        :param ordered: (optional) if the return list should be date ordered
        :param date_pattern: (optional) A pattern across the whole date range.
                If set, is the primary pattern with each subsequent pattern overriding this result
                If no other pattern is set, this will return a random date based on this pattern
        :param year_pattern: (optional) adjusts the year selection to this pattern
        :param month_pattern: (optional) adjusts the month selection to this pattern. Must be of length 12
        :param weekday_pattern: (optional) adjusts the weekday selection to this pattern. Must be of length 7
        :param hour_pattern: (optional) adjusts the hours selection to this pattern. must be of length 24
        :param minute_pattern: (optional) adjusts the minutes selection to this pattern
        :param quantity: the quantity of values that are not null. Number between 0 and 1
        :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
        :param size: the size of the sample to return. Default to 1
        :param seed: a seed value for the random function: default to None
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :return: a date or size of dates in the format given.
         """
        ordered = False if not isinstance(ordered, bool) else ordered
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        _seed = DataBuilderTools._seed() if seed is None else seed
        _dt_start = pd.to_datetime(start, errors='coerce', infer_datetime_format=True,
                                   dayfirst=day_first, yearfirst=year_first)
        _dt_until = pd.to_datetime(until, errors='coerce', infer_datetime_format=True,
                                   dayfirst=day_first, yearfirst=year_first)
        _dt_base = pd.to_datetime(default, errors='coerce', infer_datetime_format=True,
                                  dayfirst=day_first, yearfirst=year_first)
        if _dt_start is pd.NaT or _dt_until is pd.NaT:
            raise ValueError("The start or until parameters cannot be converted to a timestamp")

        # ### Apply the patterns if any ###
        rtn_dates = []
        for _ in range(size):
            _seed = DataBuilderTools._next_seed(_seed, seed)
            _min_date = (pd.Timestamp.min + pd.DateOffset(years=1)).replace(month=1, day=1, hour=0, minute=0, second=0,
                                                                            microsecond=0, nanosecond=0)
            _max_date = (pd.Timestamp.max + pd.DateOffset(years=-1)).replace(month=12, day=31, hour=23, minute=59,
                                                                             second=59, microsecond=0, nanosecond=0)
            # reset the starting base
            _dt_default = _dt_base
            if not isinstance(_dt_default, pd.Timestamp):
                _dt_default = np.random.random() * (_dt_until - _dt_start) + _dt_start
            # ### date ###
            if date_pattern is not None:
                _dp_start = DataBuilderTools._convert_date2value(_dt_start)[0]
                _dp_until = DataBuilderTools._convert_date2value(_dt_until)[0]
                value = DataBuilderTools.get_number(_dp_start, _dp_until, weight_pattern=date_pattern, seed=_seed)
                _dt_default = DataBuilderTools._convert_value2date(value)[0]
            # ### years ###
            rand_year = _dt_default.year
            if year_pattern is not None:
                rand_select = DataBuilderTools._date_choice(_dt_start, _dt_until, year_pattern, seed=_seed)
                if rand_select is pd.NaT:
                    rtn_dates.append(rand_select)
                    continue
                rand_year = rand_select.year
            _max_date = _max_date.replace(year=rand_year)
            _min_date = _min_date.replace(year=rand_year)
            _dt_default = _dt_default.replace(year=rand_year)
            # ### months ###
            rand_month = _dt_default.month
            rand_day = _dt_default.day
            if month_pattern is not None:
                month_start = _dt_start if _dt_start.year == _min_date.year else _min_date
                month_end = _dt_until if _dt_until.year == _max_date.year else _max_date
                rand_select = DataBuilderTools._date_choice(month_start, month_end, month_pattern,
                                                            limits='month', seed=_seed)
                if rand_select is pd.NaT:
                    rtn_dates.append(rand_select)
                    continue
                rand_month = rand_select.month
                rand_day = _dt_default.day if _dt_default.day <= rand_select.daysinmonth else rand_select.daysinmonth
            _max_date = _max_date.replace(month=rand_month, day=rand_day)
            _min_date = _min_date.replace(month=rand_month, day=rand_day)
            _dt_default = _dt_default.replace(month=rand_month, day=rand_day)
            # ### weekday ###
            if weekday_pattern is not None:
                if not len(weekday_pattern) == 7:
                    raise ValueError("The weekday_pattern mut be a list of size 7 with index 0 as Monday")
                _weekday = DataBuilderTools._weighted_choice(weekday_pattern, seed=_seed)
                if _weekday != _min_date.dayofweek:
                    if _dt_start <= (_dt_default + Week(weekday=_weekday)) <= _dt_until:
                        rand_day = (_dt_default + Week(weekday=_weekday)).day
                        rand_month = (_dt_default + Week(weekday=_weekday)).month
                    elif _dt_start <= (_dt_default - Week(weekday=_weekday)) <= _dt_until:
                        rand_day = (_dt_default - Week(weekday=_weekday)).day
                        rand_month = (_dt_default - Week(weekday=_weekday)).month
                    else:
                        rtn_dates.append(pd.NaT)
                        continue
            _max_date = _max_date.replace(month=rand_month, day=rand_day)
            _min_date = _min_date.replace(month=rand_month, day=rand_day)
            _dt_default = _dt_default.replace(month=rand_month, day=rand_day)
            # ### hour ###
            rand_hour = _dt_default.hour
            if hour_pattern is not None:
                hour_start = _dt_start if _min_date.strftime('%d%m%Y') == _dt_start.strftime('%d%m%Y') else _min_date
                hour_end = _dt_until if _max_date.strftime('%d%m%Y') == _dt_until.strftime('%d%m%Y') else _max_date
                rand_select = DataBuilderTools._date_choice(hour_start, hour_end, hour_pattern,
                                                            limits='hour', seed=seed)
                if rand_select is pd.NaT:
                    rtn_dates.append(rand_select)
                    continue
                rand_hour = rand_select.hour
            _max_date = _max_date.replace(hour=rand_hour)
            _min_date = _min_date.replace(hour=rand_hour)
            _dt_default = _dt_default.replace(hour=rand_hour)
            # ### minutes ###
            rand_minute = _dt_default.minute
            if minute_pattern is not None:
                minute_start = _dt_start \
                    if _min_date.strftime('%d%m%Y%H') == _dt_start.strftime('%d%m%Y%H') else _min_date
                minute_end = _dt_until \
                    if _max_date.strftime('%d%m%Y%H') == _dt_until.strftime('%d%m%Y%H') else _max_date
                rand_select = DataBuilderTools._date_choice(minute_start, minute_end, minute_pattern, seed=seed)
                if rand_select is pd.NaT:
                    rtn_dates.append(rand_select)
                    continue
                rand_minute = rand_select.minute
            _max_date = _max_date.replace(minute=rand_minute)
            _min_date = _min_date.replace(minute=rand_minute)
            _dt_default = _dt_default.replace(minute=rand_minute)
            # ### get the date ###
            _dt_default = _dt_default.replace(second=np.random.randint(60))
            if isinstance(_dt_default, pd.Timestamp):
                _dt_default = _dt_default.tz_localize(None)
            rtn_dates.append(_dt_default)
        if ordered:
            rtn_dates = sorted(rtn_dates)
        rtn_list = []
        if isinstance(date_format, str):
            for d in rtn_dates:
                if isinstance(d, pd.Timestamp):
                    rtn_list.append(d.strftime(date_format))
                else:
                    rtn_list.append(str(d))
        else:
            rtn_list = rtn_dates
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def get_names(size: int=None, mf_weighting: list=None, seed: int=None, quantity: float = None):
        """ returns a DataFrame of forename, surname and gender.

        :param size: the size of the sample, if None then set to 1
        :param mf_weighting (optional) Male (idx 0) to Female (idx 1) weighting. if None then just random
        :param quantity: the quantity of values that are not null. Number between 0 and 1
        :param seed: (optional) a seed value for the random function: default to None
        :return: a pandas DataFrame of males and females
        """
        if not isinstance(mf_weighting, list):
            mf_weighting = [1, 1]
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        _seed = DataBuilderTools._seed() if seed is None else seed
        if len(mf_weighting) != 2:
            raise ValueError("if used, mf_weighting must be a list of len 2 representing Male to female ratio")

        df = pd.DataFrame()
        df['surname'] = ProfileSample.surnames(size=size, seed=_seed)

        m_names = ProfileSample.male_names(size=size, seed=_seed)
        f_names = ProfileSample.female_names(size=size, seed=_seed)
        forename = []
        gender = []
        for idx in range(size):
            _seed = DataBuilderTools._next_seed(_seed, seed)
            pattern = DataBuilderTools._normailse_weights(mf_weighting, size=size, count=idx)
            weight_idx = DataBuilderTools._weighted_choice(pattern, seed=_seed)
            if weight_idx:
                fn = np.random.choice(f_names)
                forename.append(fn)
                f_names.remove(fn)
                gender.append('F')
                if len(f_names) < 1:
                    f_names = ProfileSample.female_names(seed=_seed)
            else:
                mn = np.random.choice(m_names)
                forename.append(mn)
                m_names.remove(mn)
                gender.append('M')
                if len(m_names) < 1:
                    m_names = ProfileSample.male_names(seed=_seed)
        df['forename'] = forename
        df['gender'] = DataBuilderTools._set_quantity(gender, quantity=quantity, seed=_seed)
        return df

    @staticmethod
    def get_file_column(labels: [str, list], filename: str, size: int=None, randomize: bool=None, seed: int=None,
                        file_format: str=None, **kwargs):
        """ gets a column or columns of data from a CSV file returning them as a Series or Dataframe
        column is requested

        :param labels: the header labels to extract
        :param filename: the file to load (must be CSV)
        :param size: (optional) the size of the sample to retrieve, if None then it assumes all
        :param randomize: (optional) if the selection should be randomised. Default is False
        :param seed: (optional) a seed value for the random function: default to None
        :param file_format: (optional) the format of the file. currently only csv and pickle supported. Default csv
        :param kwargs: (optional) any extra key word args to include in pd.read_csv() method
        :return: DataFrame or List
        """
        _seed = DataBuilderTools._seed() if seed is None else seed
        randomize = False if not isinstance(randomize, bool) else randomize
        labels = AbstractPropertiesManager.list_formatter(labels)
        if file_format == 'pickle':
            df = pd.read_pickle(filename, **kwargs)
        else:
            df = Cleaner.filter_columns(pd.read_csv(filename, **kwargs), headers=labels)
        if randomize:
            df = df.sample(frac=1, random_state=_seed).reset_index(drop=True)
        for label in labels:
            if label not in df.columns:
                raise NameError("The label '{}' could not be found in the file {}".format(label, filename))
        if not isinstance(size, int):
            size = df.shape[0]
        if df.shape[1] == 1:
            return list(df.iloc[:size, 0])
        return df.iloc[:size]

    @staticmethod
    def associate_analysis(analysis_dict: dict, size: int=None, df_ref: pd.DataFrame=None, seed: int=None):
        """ builds a set of colums based on an analysis dictionary of weighting (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param analysis_dict: the analysis dictionary (see analyse_association(...))
        :param size: (optional) the size. should be greater than or equal to the analysis sample for best results.
        :param df_ref: (optional) an already constructed df of size 'size' to take reference values from
        :param seed: seed: (optional) a seed value for the random function: default to None
        :return: a dataframe based on the association dictionary
        """
        tools = DataBuilderTools

        def get_row(analysis: dict):
            for label, values in analysis.items():
                if not values.get('analysis'):
                    raise KeyError("the lable '{}' does not have an 'analysis', key".format(label))
                dtype = values.get('analysis').get('dtype')
                selection = values.get('analysis').get('selection')
                weight_pattern = values.get('analysis').get('weighting')
                null_values = values.get('analysis').get('null_values')
                null_dict[label] = null_values
                result = tools.get_category(selection=selection, weight_pattern=weight_pattern, seed=seed, size=1)[0]
                row_dict[label] = result
                if isinstance(result, tuple) and str(dtype).startswith('num'):
                    precision = values.get('analysis').get('precision')
                    currency = values.get('analysis').get('currency')
                    (lower, upper) = result
                    # overwrite the results
                    row_dict[label] = tools.get_number(lower, upper, weight_pattern=weight_pattern, precision=precision,
                                                       currency=currency, seed=seed, size=1)[0]
                if isinstance(result, tuple) and str(dtype).startswith('date'):
                    date_format = values.get('analysis').get('date_format')
                    day_first = values.get('analysis').get('day_first')
                    year_first = values.get('analysis').get('year_first')
                    year_pattern = values.get('analysis').get('year_pattern')
                    month_pattern = values.get('analysis').get('month_pattern')
                    weekday_pattern = values.get('analysis').get('weekday_pattern')
                    hour_pattern = values.get('analysis').get('hour_pattern')
                    minute_pattern = values.get('analysis').get('minute_pattern')
                    (lower, upper) = result
                    # overwrite the results
                    row_dict[label] = tools.get_datetime(start=lower, until=upper, date_pattern=weight_pattern,
                                                         year_pattern=year_pattern, month_pattern=month_pattern,
                                                         weekday_pattern=weekday_pattern, hour_pattern=hour_pattern,
                                                         minute_pattern=minute_pattern, date_format=date_format,
                                                         size=1, seed=seed, day_first=day_first, year_first=year_first)
                if values.get('sub_category'):
                    next_item = values.get('sub_category').get(result)
                    get_row(next_item)
            return

        size = 1 if not isinstance(size, int) else size
        size = df_ref.shape[0] if isinstance(df_ref, pd.DataFrame) else size
        rtn_df = pd.DataFrame()
        for index in range(size):
            row_dict = {}
            null_dict = {}
            get_row(analysis_dict)
            rtn_df = rtn_df.append([row_dict], ignore_index=True)
        return rtn_df

    @staticmethod
    def associate_dataset(dataset: Any, associations: list, actions: dict, default_value: Any=None,
                          default_header: str=None, day_first: bool=True, quantity:  float=None, seed: int=None):
        """ Associates a a -set of criteria of an input values to a set of actions
            The association dictionary takes the form of a set of dictionaries in a list with each item in the list
            representing an index key for the action dictionary. Each dictionary are to associated relationship.
            In this example for the first index the associated values should be header1 is within a date range
            and header2 has a value of 'M'
                association = [{'header1': {'expect': 'date',
                                            'value': ['12/01/1984', '14/01/2014']},
                                'header2': {'expect': 'category',
                                            'value': ['M']}},
                                {...}]

            if the dataset is not a DataFrame then the header should be omitted. in this example the association is
            a range comparison between 2 and 7 inclusive.
                association= [{'expect': 'number', 'value': [2, 7]},
                              {...}]

            The actions dictionary takes the form of an index referenced dictionary of actions, where the key value
            of the dictionary corresponds to the index of the association list. In other words, if a match is found
            in the association, that list index is used as reference to the action to execute.
                {0: {'action': '', 'kwargs' : {}},
                 1: {...}}
            you can also use the action to specify a specific value:
                {0: {'action': ''},
                 1: {'action': ''}}

        :param dataset: the dataset to map against, this can be a str, int, float, list, Series or DataFrame
        :param associations: a list of categories (can also contain lists for multiple references.
        :param actions: the correlated set of categories that should map to the index
        :param default_header: (optional) if no association, the default column header to take the value from.
                    if None then the default_value is taken.
                    Note for non-DataFrame datasets the default header is '_default'
        :param default_value: (optional) if no default header then this value is taken if no association
        :param day_first: (optional) if expected type is date, indicates if the day is first. Default to true
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
        :return: a list of equal length to the one passed
        """
        quantity = DataBuilderTools._quantity(quantity)
        _seed = DataBuilderTools._seed() if seed is None else seed

        if not isinstance(dataset, (str, int, float, list, pd.Series, pd.DataFrame)):
            raise TypeError("The parameter values is not an accepted type")
        if not isinstance(associations, (list, dict)):
            raise TypeError("The parameter reference must be a list or dict")
        _dataset = dataset
        _associations = associations
        if isinstance(_dataset, (str, int, float)):
            _dataset = AbstractPropertiesManager.list_formatter(_dataset)
        if isinstance(_dataset, (list, pd.Series)):
            tmp = pd.DataFrame()
            tmp['_default'] = _dataset
            _dataset = tmp
            tmp = []
            for item in _associations:
                tmp.append({'_default': item})
            _associations = tmp
        if not isinstance(_dataset, pd.DataFrame):
            raise TypeError("The dataset given is not or could not be convereted to a pandas DataFrame")
        class_methods = DataBuilderTools().__dir__()

        rtn_list = []
        for index in range(_dataset.shape[0]):
            action_idx = None
            for idx in range(len(_associations)):
                associate_dict = _associations[idx]
                is_match = [0] * len(associate_dict.keys())
                match_idx = 0
                for header, lookup in associate_dict.items():
                    df_value = _dataset[header].iloc[index]
                    expect = lookup.get('expect')
                    chk_value = AbstractPropertiesManager.list_formatter(lookup.get('value'))
                    if expect.lower() in ['number', 'n']:
                        if len(chk_value) == 1:
                            [s] = [e] = chk_value
                        else:
                            [s, e] = chk_value
                        if s <= df_value <= e:
                            is_match[match_idx] = True
                    elif expect.lower() in ['date', 'datetime', 'd']:
                        [s, e] = chk_value
                        value_date = pd.to_datetime(df_value, errors='coerce', infer_datetime_format=True,
                                                    dayfirst=day_first)
                        s_date = pd.to_datetime(s, errors='coerce', infer_datetime_format=True, dayfirst=day_first)
                        e_date = pd.to_datetime(e, errors='coerce', infer_datetime_format=True, dayfirst=day_first)
                        if value_date is pd.NaT or s_date is pd.NaT or e_date is pd.NaT:
                            break
                        if s_date <= value_date <= e_date:
                            is_match[match_idx] = True
                    elif expect.lower() in ['category', 'c']:
                        if df_value in chk_value:
                            is_match[match_idx] = True
                    else:
                        break
                    match_idx += 1
                if all(x for x in is_match):
                    action_idx = idx
                    break
            if action_idx is None or actions.get(action_idx) is None:
                if default_header is not None and default_header in _dataset.columns:
                    rtn_list.append(_dataset[default_header].iloc[index])
                else:
                    rtn_list.append(default_value)
                continue
            method = actions.get(action_idx).get('action')
            if method is None:
                raise ValueError("There is no 'action' key at index [{}]".format(action_idx))
            if method in class_methods:
                kwargs = actions.get(action_idx).get('kwargs').copy()
                for k, v in kwargs.items():
                    if isinstance(v, dict) and '_header' in v.keys():
                        if v.get('_header') not in dataset.columns:
                            raise ValueError("Dataset header '{}' does not exist: see action: {} -> key: {}".format(
                                v.get('_header'), action_idx, k))
                        kwargs[k] = dataset[v.get('_header')].iloc[index]
                result = eval("DataBuilderTools.{}(**{})".format(method, kwargs).replace('nan', 'None'))
                if isinstance(result, list):
                    if len(result) > 0:
                        rtn_list.append(result.pop())
                    else:
                        rtn_list.append(None)
                else:
                    rtn_list.append(result)
            else:
                if isinstance(method, dict):
                    if method.get('_header') not in dataset.columns:
                        raise ValueError("Dataset header '{}' does not exist: see action: {} -> key: action".format(
                            method.get('_header'), action_idx))
                    rtn_list.append(dataset[method.get('_header')].iloc[index])
                else:
                    rtn_list.append(method)
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def associate_custom(df: pd.DataFrame, action: str, use_exec: bool=False, **kwargs):
        """ enacts an action on a dataFrame, returning the output of the action or the DataFrame if using exec or
        the evaluation returns None. Not that if using the input dataframe in your action, it is internally referenced
        as it's parameter name 'df'.

        :param df: a pd.DataFrame used in the action
        :param action: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :return: a list or pandas.DataFrame
        """
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'df' not in local_kwargs:
            local_kwargs['df'] = df

        result = exec(action, globals(), local_kwargs) if use_exec else eval(action, globals(), local_kwargs)
        if result is None:
            return df
        return result

    @staticmethod
    def correlate_numbers(values: Any, spread: float=None, offset: float=None, action: str=None, precision: int=None,
                          fill_nulls: bool=None, quantity: float=None, seed: int=None, keep_zero: bool=None,
                          min_value: [int, float]= None, max_value: [int, float]= None):
        """ returns a number that correlates to the value given. The spread is based on a normal distribution
        with the value being the mean and the spread its standard deviation from that mean

        :param values: a single value or list of values to correlate
        :param spread: (optional) the random spread or deviation from the value. defaults to 0
        :param offset: (optional) how far from the value to offset. defaults to zero
        :param action: (optional) what action on the offset. Options are: 'add'(default),'multiply'
        :param precision: (optional) how many decimal places. default to 3
        :param fill_nulls: (optional) if True then fills nulls with the most common values
        :param quantity: (optional) a number between 0 and 1 preresenting the percentage quantity of the data
        :param seed: (optional) the random seed. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
        :param min_value: a minimum value not to go below
        :param max_value: a max value not to go above
        :return: an equal length list of correlated values
        """
        offset = 0.0 if offset is None else offset
        spread = 0.0 if spread is None else spread
        precision = 3 if precision is None else precision
        action = 'add' if not isinstance(action, str) else action
        keep_zero = False if not isinstance(keep_zero, bool) else True
        fill_nulls = False if fill_nulls is None or not isinstance(fill_nulls, bool) else fill_nulls
        quantity = DataBuilderTools._quantity(quantity)
        _seed = DataBuilderTools._seed() if seed is None else seed

        values = AbstractPropertiesManager.list_formatter(values)

        if values is None or len(values) == 0:
            return list()
        mode_choice = DataBuilderTools._mode_choice(values) if fill_nulls else list()
        rtn_list = []
        for index in range(len(values)):
            if keep_zero and values[index] == 0:
                rtn_list.append(0)
                continue
            next_index = False
            counter = 0
            while not next_index:
                counter += 1
                if counter > 100:
                    raise ValueError("The minimum or maximum values are too constraining to correlate numbers")
                v = values[index]
                _seed = DataBuilderTools._next_seed(_seed, seed)
                if fill_nulls and len(mode_choice) > 0 and (str(v) == 'nan' or not isinstance(v, (int, float))):
                    v = int(np.random.choice(mode_choice))
                if isinstance(v, (int, float)):
                    v = v * offset if action == 'multiply' else v + offset
                    _result = round(np.random.normal(loc=v, scale=spread), precision)
                    if precision == 0:
                        _result = int(_result)
                else:
                    _result = v
                if isinstance(min_value, (int, float)) and _result < min_value:
                    continue
                elif isinstance(max_value, (int, float)) and _result > max_value:
                    continue
                rtn_list.append(_result)
                next_index = True
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @ staticmethod
    def correlate_categories(values: Any, correlations: list, actions: dict, value_type: str,
                             day_first: bool = True, quantity:  float=None, seed: int=None):
        """ correlation of a set of values to an action, the correlations must map to the dictionary index values.
        Note. to use the current value in the passed values as a parameter value pass an empty dict {} as the keys
        value. If you want the action value to be the current value of the passed value then again pass an empty dict
        action to be the current value
            simple correlation list:
                ['A', 'B', 'C'] # if values is 'A' then action is 0 and so on
            multiple choice correlation
                [['A','B'], 'C'] # if values is 'A' OR 'B' then action is 0 and so on
            actions dictionary where the action is a class method name and kwargs its parameters
                {0: {'action': '', 'kwargs' : {}}, 1: {'action': '', 'kwargs' : {}}}
            you can also use the action to specify a specific value:
                {0: {'action': ''}, 1: {'action': ''}}

        :param values: the category values to map against
        :param correlations: a list of categories (can also contain lists for multiple correlations.
        :param actions: the correlated set of categories that should map to the index
        :param value_type: the type found in the values (options are 'category' ('c'), 'number', or 'datetime'('date'))
        :param day_first: (optional) if type is date indictes if the day is first. Default to true
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: a seed value for the random function: default to None
        :return: a list of equal length to the one passed
        """
        quantity = DataBuilderTools._quantity(quantity)
        _seed = DataBuilderTools._seed() if seed is None else seed
        if value_type.lower() not in ['c', 'n', 'd', 'category', 'number', 'datetime', 'date']:
            raise ValueError("the category type must be one of C, N, D or Category, Number, Datetime/Date")
        corr_list = []
        for corr in correlations:
            corr_list.append(AbstractPropertiesManager.list_formatter(corr))
        if values is None or len(values) == 0:
            return list()
        class_methods = DataBuilderTools().__dir__()

        rtn_list = []
        for value_index in range(len(values)):
            value = values[value_index]
            _seed = DataBuilderTools._next_seed(_seed, seed)
            corr_index = None
            for i in range(len(corr_list)):
                if value_type.lower() in ['number', 'n']:
                    if not isinstance(value, (float, int)):
                        break
                    if len(corr_list[i]) == 1:
                        [s] = [e] = corr_list[i]
                    else:
                        [s, e] = corr_list[i]
                    if s <= value <= e:
                        corr_index = i
                        break
                elif value_type.lower() in ['date', 'datetime', 'd']:
                    [s, e] = corr_list[i]
                    value_date = pd.to_datetime(value, errors='coerce', infer_datetime_format=True, dayfirst=day_first)
                    s_date = pd.to_datetime(s, errors='coerce', infer_datetime_format=True, dayfirst=day_first)
                    e_date = pd.to_datetime(e, errors='coerce', infer_datetime_format=True, dayfirst=day_first)
                    if value_date is pd.NaT or s_date is pd.NaT or e_date is pd.NaT:
                        break
                    if s <= value <= e:
                        corr_index = i
                        break
                else:
                    if value in corr_list[i]:
                        corr_index = i
                        break
            if corr_index is None or actions.get(corr_index) is None:
                rtn_list.append(value)
                continue
            method = actions.get(corr_index).get('action')
            if method is None:
                raise ValueError("There is no 'action' key at index [{}]".format(corr_index))
            if method in class_methods:
                kwargs = actions.get(corr_index).get('kwargs').copy()
                for k, v in kwargs.items():
                    if isinstance(v, dict):
                        kwargs[k] = value
                result = eval("DataBuilderTools.{}(**{})".format(method, kwargs))
                if isinstance(result, list):
                    if len(result) > 0:
                        rtn_list.append(result.pop())
                    else:
                        rtn_list.append(None)
                else:
                    rtn_list.append(result)
            else:
                if isinstance(method, dict):
                    method = value
                rtn_list.append(method)
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def correlate_dates(dates: Any, offset: [int, dict]=None, date_format: str=None, lower_spread: [int, dict]=None,
                        upper_spread: [int, dict]=None, ordered: bool=None, date_pattern: list = None,
                        year_pattern: list = None, month_pattern: list = None, weekday_pattern: list = None,
                        hour_pattern: list = None, minute_pattern: list = None, min_date: str=None,
                        max_date: str=None, fill_nulls: bool=None, day_first: bool=True, year_first: bool=False,
                        quantity: float = None, seed: int=None):
        """ correlates dates to an existing date or list of dates.

        :param dates: the date or set of dates to correlate
        :param offset:  (optional)and offset to the date. if int then assumed a 'years' offset
                int or dictionary associated with pd.DateOffset(). eg {'months': 1, 'days': 5}
        :param lower_spread: (optional) the lower boundary from the relative date. if int then assume 'days' spread
                int or dictionary associated with pd.DateOffset(). eg {'days': 2}
        :param upper_spread: (optional) the upper boundary from the relative date. if int then assume 'days' spread
                int or dictionary associated with pd.DateOffset(). eg {'hours': 7}
        :param ordered: (optional) if the return list should be date ordered
        :param date_pattern: (optional) A pattern across the whole date range.
                If set, is the primary pattern with each subsequent pattern overriding this result
                If no other pattern is set, this will return a random date based on this pattern
        :param year_pattern: (optional) adjusts the year selection to this pattern
        :param month_pattern: (optional) adjusts the month selection to this pattern. Must be of length 12
        :param weekday_pattern: (optional) adjusts the weekday selection to this pattern. Must be of length 7
        :param hour_pattern: (optional) adjusts the hours selection to this pattern. must be of length 24
        :param minute_pattern: (optional) adjusts the minutes selection to this pattern
        :param min_date: (optional)a minimum date not to go below
        :param max_date: (optional)a max date not to go above
        :param fill_nulls: (optional) if no date values should remain untouched or filled based on the list mode date
        :param day_first: (optional) if the dates given are day first firmat. Default to True
        :param year_first: (optional) if the dates given are year first. Default to False
        :param date_format: (optional) the format of the output
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
        :return: a list of equal size to that given
        """
        quantity = DataBuilderTools._quantity(quantity)
        _seed = DataBuilderTools._seed() if seed is None else seed
        fill_nulls = False if fill_nulls is None or not isinstance(fill_nulls, bool) else fill_nulls
        if date_format is None:
            date_format = '%d-%m-%YT%H:%M:%S'

        offset = {} if offset is None or not isinstance(offset, (int, dict)) else offset
        offset = {'years': offset} if isinstance(offset, int) else offset
        lower_spread = {} if lower_spread is None or not isinstance(lower_spread, (int, dict)) else lower_spread
        upper_spread = {} if upper_spread is None or not isinstance(upper_spread, (int, dict)) else upper_spread
        lower_spread = {'days': lower_spread} if isinstance(lower_spread, int) else lower_spread
        upper_spread = {'days': upper_spread} if isinstance(upper_spread, int) else upper_spread

        def _clean(control):
            _unit_type = ['years', 'months', 'weeks', 'days', 'leapdays', 'hours', 'minutes', 'seconds']
            _params = {}
            if isinstance(control, int):
                return {'years': control}
            if isinstance(control, dict):
                for k, v in control.items():
                    if k in _unit_type:
                        _params[k] = v
            return _params

        _min_date = pd.to_datetime(min_date, errors='coerce', infer_datetime_format=True,
                                   dayfirst=day_first, yearfirst=year_first)
        if _min_date is None or _min_date is pd.NaT:
            _min_date = pd.Timestamp.min

        _max_date = pd.to_datetime(max_date, errors='coerce', infer_datetime_format=True,
                                   dayfirst=day_first, yearfirst=year_first)
        if _max_date is None or _max_date is pd.NaT:
            _max_date = pd.Timestamp.max

        if _min_date >= _max_date:
            raise ValueError("the min_date {} must be less than max_date {}".format(min_date, max_date))

        dates = AbstractPropertiesManager.list_formatter(dates)
        if dates is None or len(dates) == 0:
            return list()
        mode_choice = DataBuilderTools._mode_choice(dates) if fill_nulls else list()
        rtn_list = []
        for d in dates:
            _seed = DataBuilderTools._next_seed(_seed, seed)
            if fill_nulls and len(mode_choice) > 0 and not isinstance(d, str):
                d = int(np.random.choice(mode_choice))
            _control_date = pd.to_datetime(d, errors='coerce', infer_datetime_format=True,
                                           dayfirst=day_first, yearfirst=year_first)
            if isinstance(_control_date, pd.Timestamp):
                _offset_date = _control_date + pd.DateOffset(**_clean(offset))
                if _max_date <= _offset_date <= _min_date:
                    err_date = _offset_date.strftime(date_format)
                    raise ValueError(
                        "The offset_date {} is does not fall between the min and max dates".format(err_date))
                _upper_spread_date = _offset_date + pd.DateOffset(**_clean(upper_spread))
                _lower_spread_date = _offset_date - pd.DateOffset(**_clean(lower_spread))
                _result = None
                while not _result:
                    sample_list = DataBuilderTools.get_datetime(start=_lower_spread_date, until=_upper_spread_date,
                                                                ordered=ordered,
                                                                date_pattern=date_pattern, year_pattern=year_pattern,
                                                                month_pattern=month_pattern,
                                                                weekday_pattern=weekday_pattern,
                                                                hour_pattern=hour_pattern,
                                                                minute_pattern=minute_pattern, seed=_seed)
                    _sample_date = sample_list[0]
                    if _sample_date is None or _sample_date is pd.NaT:
                        raise ValueError("Unable to generate a random datetime, {} returned".format(sample_list))
                    if not _min_date <= _sample_date <= _max_date:
                        _result = None
                    else:
                        _result = _sample_date.strftime(date_format) if isinstance(date_format, str) else _sample_date
            else:
                _result = d
            rtn_list.append(_result)
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    @staticmethod
    def unique_identifiers(from_value: int, to_value: int=None, size: int=None, prefix: str=None, suffix: str=None,
                           weight_pattern: list = None, quantity: float=None, seed: int=None):
        """ returns a list of unique identifiers randomly selected between the from_value and to_value

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param size: the size of the sample. Must be smaller than the range
        :param prefix: a prefix to the number . Default to nothing
        :param suffix: a suffix to the number. default to nothing
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param quantity: a number between 0 and 1 preresenting the percentage quantity of the data
        :param seed: a seed value for the random function: default to None
        :return: a unique identifer randomly selected from the range
        """
        (from_value, to_value) = (0, from_value) if not isinstance(to_value, (float, int)) else (from_value, to_value)
        quantity = DataBuilderTools._quantity(quantity)
        size = 1 if size is None else size
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        rtn_list = []
        for i in DataBuilderTools.unique_numbers(start=from_value, until=to_value, size=size,
                                                 weight_pattern=weight_pattern, seed=seed):
            rtn_list.append("{}{}{}".format(prefix, i, suffix))
        return DataBuilderTools._set_quantity(rtn_list, quantity=quantity, seed=seed)

    @staticmethod
    def unique_numbers(start: [int, float], until: [int, float], size: int, weight_pattern: list=None,
                       precision: int=None, seed: int=None) -> list:
        """Generate a number tokens of a specified length.

        :param start: the start number
        :param until: then end boundary
        :param size: The number of tokens to return
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param seed: a seed value for the random function: default to None
        """
        if until - start <= size:
            raise ValueError("The number of tokens must be less than the the number pool")
        _seed = DataBuilderTools._seed() if seed is None else seed
        seen = set()
        add = seen.add
        attempts = 0
        max_attempts = 10
        while len(seen) < size:
            if attempts > max_attempts:
                raise InterruptedError(
                    "Unique Date Sequence stopped: After {} attempts unable to create unique set".format(max_attempts))
            for token in DataBuilderTools.get_number(from_value=start, to_value=until, weight_pattern=weight_pattern,
                                                     precision=precision, size=size+10, seed=_seed):
                add(token)
                if len(seen) == size:
                    break
            _seed = DataBuilderTools._next_seed(_seed, seed)
            attempts += 1
        return list(seen)

    @staticmethod
    def unique_date_seq(start: Any, until: Any, default: Any = None, ordered: bool=None,
                        date_pattern: list = None, year_pattern: list = None, month_pattern: list = None,
                        weekday_pattern: list = None, hour_pattern: list = None, minute_pattern: list = None,
                        date_format: str = None, size: int = None, seed: int = None,
                        day_first: bool = True, year_first: bool = False):
        """creates an ordered and unique date sequence based on the parameters passed

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param default: (optional) a fixed starting date that patterns are applied too.
        :param ordered: (optional) if the return list should be date ordered
        :param date_pattern: (optional) A pattern across the whole date range.
                If set, is the primary pattern with each subsequent pattern overriding this result
                If no other pattern is set, this will return a random date based on this pattern
        :param year_pattern: (optional) adjusts the year selection to this pattern
        :param month_pattern: (optional) adjusts the month selection to this pattern. Must be of length 12
        :param weekday_pattern: (optional) adjusts the weekday selection to this pattern. Must be of length 7
        :param hour_pattern: (optional) adjusts the hours selection to this pattern. must be of length 24
        :param minute_pattern: (optional) adjusts the minutes selection to this pattern
        :param date_format: the format of the date to be returned. default '%d-%m-%Y'
        :param size: the size of the sample to return. Default to 1
        :param seed: a seed value for the random function: default to None
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :return: a date or size of dates in the format given.
        """
        size = 1 if size is None or not isinstance(size, int) else size
        _seed = DataBuilderTools._seed() if seed is None else seed
        seen = set()
        add = seen.add
        attempt = 0
        max_attempts = 5
        while len(seen) < size:
            if attempt > max_attempts:
                raise InterruptedError(
                    "Unique Date Sequence stopped: After {} attempts unable to create unique set".format(max_attempts))
            for token in DataBuilderTools.get_datetime(start=start, until=until, default=default, ordered=ordered,
                                                       date_pattern=date_pattern, year_pattern=year_pattern,
                                                       month_pattern=month_pattern, weekday_pattern=weekday_pattern,
                                                       hour_pattern=hour_pattern, minute_pattern=minute_pattern,
                                                       date_format=date_format, day_first=day_first,
                                                       year_first=year_first, size=size+10, seed=_seed):
                add(token)
                if len(seen) == size:
                    break
            _seed = DataBuilderTools._next_seed(_seed, seed)
            attempt += 1
        return list(seen)

    @staticmethod
    def unique_str_tokens(length: int=None, size: int=None, pool: str=None) -> list:
        """Generate a list of str tokens of a specified length.

        :param length: Length of each token
        :param size: number of tokens or size of sample
        :param pool: Iterable of characters to choose from
        """
        if not isinstance(pool, str):
            pool = string.ascii_letters
        if not isinstance(size, int):
            size = np.random.randint(1, 5)
        seen = set()
        add = seen.add
        attempts = 0
        max_attempts = size + size * size
        while len(seen) < size:
            if attempts > max_attempts:
                raise InterruptedError(
                    "After {} attempts unable to create unique set. "
                    "The pool might not be big enough for the length and size to be unique".format(max_attempts))
            if length == 0:
                length = np.random.randint(2, 8)
            token = ''.join(np.random.choice(list(pool), size=length))
            add(token)
            attempts += 1
        return list(seen)

    @staticmethod
    def analyse_date(values: Any, granularity: [int, float, pd.Timedelta]=None, lower: Any = None, upper: Any = None,
                     day_first: bool = True, year_first: bool = False, date_format: str = None, chunk_size: int = None,
                     replace_zero: [int, float] = None):
        """Analyses a set of dates and returns a dictionary of selection and weighting

        :param values: the values to analyse
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - the number of sections to break the value range into
                pd.Timedelta passed - a frequency time delta
        :param lower: (optional) the lower limit of the number value. Takes min() if not set
        :param upper: (optional) the upper limit of the number value. Takes max() if not set
        :param day_first: if the date provided has day first
        :param year_first: if the date provided has year first
        :param date_format: the format of the output dates, if None then pd.Timestamp
        :param chunk_size: (optional) number of chuncks if you want weighting over the length of the dataset
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :return: a dictionary of results
        """
        values = pd.to_datetime(values, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        values = mdates.date2num(values)
        values = pd.Series(values)
        lower = pd.to_datetime(lower, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                               yearfirst=year_first)
        upper = pd.to_datetime(upper, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                               yearfirst=year_first)
        lower = values.min() if not isinstance(lower, pd.Timestamp) else mdates.date2num(lower)
        upper = values.max() if not isinstance(upper, pd.Timestamp) else mdates.date2num(upper)
        if lower < upper:
            if isinstance(granularity, pd.Timedelta):
                granularity = mdates.date2num(mdates.num2date(lower) + granularity) - lower
            rtn_dict = DataBuilderTools.analyse_number(values, granularity=granularity, lower=lower, upper=upper,
                                                       chunk_size=chunk_size, replace_zero=replace_zero, precision=10)
        else:
            value_size = values.size
            values = values.dropna()
            null_values = np.round(((value_size - values.size) / value_size) * 100, 2) if value_size > 0 else 0
            rtn_dict = {'selection': [(lower, upper)], 'weighting': [100 - null_values], 'lower': lower, 'upper': upper,
                        'granularity': granularity, 'dtype': 'date', 'quantity': null_values, 'sample': [values.size]}
        # tidy back all the dates
        rtn_dict['selection'] = [(tuple([pd.Timestamp(mdates.num2date(y)) for y in x])) for x in rtn_dict['selection']]
        rtn_dict['lower'] = pd.Timestamp(mdates.num2date(rtn_dict['lower']))
        rtn_dict['upper'] = pd.Timestamp(mdates.num2date(rtn_dict['upper']))
        if isinstance(date_format, str):
            rtn_dict['selection'] = [(tuple([y.strftime(date_format) for y in x])) for x in rtn_dict['selection']]
            rtn_dict['lower'] = rtn_dict['lower'].strftime(date_format)
            rtn_dict['upper'] = rtn_dict['upper'].strftime(date_format)
        rtn_dict['dtype'] = 'date'
        return rtn_dict

    @staticmethod
    def analyse_number(values: Any, granularity: [int, float]=None, lower: [int, float]=None, upper: [int, float]=None,
                       chunk_size: int=None, replace_zero: [int, float]=None, precision: int=None):
        """Analyses a set of values and returns a dictionary of selection and weighting

        :param values: the values to analyse
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - represents the number of periods
                float passed - the length of each interval
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param chunk_size: (optional) number of chuncks if you want weighting over the length of the dataset
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param precision: (optional) by default set to 3.
        :return: a dictionary of results
        """
        values = pd.Series(values)
        precision = 3 if not isinstance(precision, int) else precision
        granularity = 3 if not isinstance(granularity, (int, float)) else granularity
        replace_zero = 0 if not isinstance(replace_zero, (int, float)) else replace_zero
        chunk_size = 1 if not isinstance(chunk_size, int) or chunk_size > values.size else chunk_size
        lower = values.min() if not isinstance(lower, (int, float)) or lower > values.min() else lower
        upper = values.max() if not isinstance(upper, (int, float)) or upper < values.max() else upper
        if lower >= upper:
            value_size = values.size
            values = values.dropna()
            null_values = np.round(((value_size - values.size) / value_size) * 100, 2) if value_size > 0 else 0
            return {'selection': [(lower, upper)], 'weighting': [100-null_values], 'lower': lower, 'upper': upper,
                    'granularity': granularity, 'dtype': 'number', 'null_values': null_values, 'sample': [values.size]}
        # get the intervals so the interval range is fixed across the chunks
        freq = granularity if isinstance(granularity, float) else None
        periods = granularity if isinstance(granularity, int) else None
        # ensure the frequency is equal to or higher than the upper value
        _upper = upper + freq - (upper % freq) if periods is None else upper
        interval_range = pd.interval_range(start=lower, end=_upper, periods=periods, freq=freq, closed='both')
        interval_range = interval_range.drop_duplicates()
        weighting = []
        null_values = []
        sample = []
        for chunk in np.array_split(values, chunk_size):
            value_size = chunk.size
            chunk = chunk.dropna()
            null_values.append(np.round(((value_size - chunk.size) / value_size) * 100, 2) if value_size > 0 else 0)
            sample.append(chunk.size)
            chunk_weights = [0] * len(interval_range)
            for v in chunk:
                chunk_weights[interval_range.get_loc(v)[-1]] += 1
            chunk_weights = pd.Series(chunk_weights)
            if value_size > 0:
                chunk_weights = chunk_weights.apply(lambda x: np.round((x / value_size) * 100, 2))
            weighting.append(chunk_weights.replace(0, replace_zero).tolist())
        if len(weighting) == 1:
            weighting = weighting[0]
        selection = interval_range.to_tuples().tolist()
        selection = [(tuple([round(y, precision) if isinstance(y, (int, float)) else y for y in x])) for x in selection]
        rtn_dict = {'selection': selection, 'weighting': weighting, 'lower': lower, 'upper': upper,
                    'granularity': granularity, 'dtype': 'number', 'null_values': null_values, 'sample': sample}
        return rtn_dict

    @staticmethod
    def analyse_category(categories: Any, chunk_size: int=None, replace_zero: [int, float]=None, nulls_list: list=None):
        """Analyses a set of categories and returns a dictionary of selection and weighting
        the return is in the form:
                {'selection': [], 'weighting': []}

        :param categories: the categories to analyse
        :param chunk_size: (optional) number of chuncks if you want weighting over the length of the dataset
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param nulls_list: (optional) a list of nulls if more than the default empty string
        :return: a dictionary of results
        """
        categories = pd.Series(categories)
        nulls_list = [''] if not isinstance(nulls_list, list) else nulls_list
        replace_zero = 0 if not isinstance(replace_zero, (int, float)) else replace_zero
        chunk_size = 1 if not isinstance(chunk_size, int) or chunk_size > categories.size else chunk_size
        selection = pd.Series(data=0, index=categories.replace(nulls_list, np.nan).dropna().unique())
        weighting = []
        null_values = []
        sample = []
        for chunk in np.array_split(categories, chunk_size):
            cat_size = chunk.size
            chunk = chunk.replace(nulls_list, np.nan).dropna()
            null_values.append(np.round(((cat_size - chunk.size) / cat_size) * 100, 2) if cat_size > 0 else 0)
            sample.append(chunk.size)
            value_count = chunk.value_counts()
            if value_count.sum() > 0:
                value_count = value_count.apply(lambda x: np.round((x / value_count.sum()) * 100, 2))
            value_count = pd.Series(selection.index.map(value_count))
            weighting.append(value_count.replace(np.nan, 0).replace(0, replace_zero).tolist())
        if len(weighting) == 1:
            weighting = weighting[0]
        rtn_dict = {'selection': categories.replace(nulls_list, np.nan).dropna().unique().tolist(),
                    'weighting': weighting, 'dtype': 'category', 'null_values': null_values, 'sample': sample}
        return rtn_dict

    @staticmethod
    def analyse_association(df: pd.DataFrame, columns_list: list, exclude_associate: list=None):
        """ Analyses the association of Category against Values and returns a dictionary of resulting weighting
        the structure of the columns_list is a list of dictionaries with the key words
            - label: the label or name of the header in the DataFrame
            - type: onr of category|number|date indicating the origin of the data, Default is 'category' if omitted
            - chunk_size: if the weighting pattern is over the size of the data the number of chunks
            - replace_zero: if a zero reference is returned it can optionally be replaced with a low probability
        and example structure might look like:
            [{'label1': {'type': 'category|number|date', 'chunk_size': int, 'replace_zero': int|float}},
             {'label2': {}}]

        :param df: the dataframe to take the columns from
        :param columns_list: a dictionary structure of collumns to select for association
        :param exclude_associate: (optional) a list of dot separated tree of items to exclude from iteration (e.g. [age.
        :return: a dictionary of association weighting
        """
        tools = DataBuilderTools

        def _get_weights(_df: pd.DataFrame, columns: list, index: int, weighting: dict, parent: list):
            for label, kwargs in columns[index].items():
                tree = parent.copy()
                tree.append(label)
                if '.'.join(tree) in exclude_associate:
                    continue
                section = {'associate': str('.'.join(tree))}
                if label not in _df.columns:
                    raise ValueError("header '{}' not found in the Dataframe".format(label))
                dtype = kwargs.get('dtype')
                chunk_size = kwargs.get('chunk_size')
                replace_zero = kwargs.get('replace_zero')
                if (dtype is None and df[label].dtype in [int, float]) or str(dtype).lower().startswith('number'):
                    granularity = kwargs.get('granularity')
                    lower = kwargs.get('lower')
                    upper = kwargs.get('upper')
                    precision = kwargs.get('precision')
                    section['analysis'] = tools.analyse_number(_df[label], granularity=granularity, lower=lower,
                                                               upper=upper, chunk_size=chunk_size,
                                                               replace_zero=replace_zero, precision=precision)
                elif str(dtype).lower().startswith('date'):
                    granularity = kwargs.get('granularity')
                    lower = kwargs.get('lower')
                    upper = kwargs.get('upper')
                    day_first = kwargs.get('day_first')
                    year_first = kwargs.get('year_first')
                    date_format = kwargs.get('date_format')
                    section['analysis'] = tools.analyse_date(_df[label], granularity=granularity, lower=lower,
                                                             upper=upper, day_first=day_first, year_first=year_first,
                                                             chunk_size=chunk_size, replace_zero=replace_zero,
                                                             date_format=date_format)
                else:
                    nulls_list = kwargs.get('nulls_list')
                    section['analysis'] = tools.analyse_category(_df[label], chunk_size=chunk_size,
                                                                 replace_zero=replace_zero, nulls_list=nulls_list)
                for category in section.get('analysis').get('selection'):
                    if section.get('sub_category') is None:
                        section['sub_category'] = {}
                    section.get('sub_category').update({category: {}})
                    sub_category = section.get('sub_category').get(category)
                    if index < len(columns) - 1:
                        if isinstance(category, tuple):
                            interval = pd.Interval(left=category[0], right=category[1])
                            df_filter = _df.loc[_df[label].apply(lambda x: x in interval)]
                        else:
                            df_filter = _df[_df[label] == category]
                        _get_weights(df_filter, columns=columns, index=index + 1, weighting=sub_category, parent=tree)
                    # tidy empty sub categories
                    if section.get('sub_category').get(category) == {}:
                        section.pop('sub_category')
                weighting[label] = section
            return

        exclude_associate = list() if not isinstance(exclude_associate, list) else exclude_associate
        rtn_dict = {}
        _get_weights(df, columns=columns_list, index=0, weighting=rtn_dict, parent=list())
        return rtn_dict

    @staticmethod
    def _convert_date2value(dates: Any, day_first: bool = True, year_first: bool = False):
        values = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        return mdates.date2num(pd.Series(values)).tolist()

    @staticmethod
    def _convert_value2date(values: [int, float], date_format: str = None):
        dates = []
        for date in mdates.num2date(values):
            date = pd.Timestamp(date)
            if isinstance(date_format, str):
                date = date.strftime(date_format)
            dates.append(date)
        return dates

    @staticmethod
    def _date_choice(start: pd.Timestamp, until: pd.Timestamp, weight_pattern: list, limits: str=None, seed: int=None):
        """ Utility method to choose a random date between two dates based on a pattern.

        :param start: the start boundary
        :param until: the boundary to go up to
        :param weight_pattern: The weight pattern to apply to the range selection
        :param limits: (optional) time units that have pattern limits
        :param seed: (optional) a seed value for the random function: default to None
        :return: a choice from the range
        """
        seed = DataBuilderTools._seed() if seed is None else seed
        np.random.seed(seed)
        diff = (until-start).days
        freq = 'Y' if diff > 4000 else 'M' if diff > 500 else 'D' if diff > 100 else 'H' if diff > 2 else 'T'
        date_range = pd.date_range(start, until, freq=freq)
        _pattern = weight_pattern
        if limits in ['month', 'hour']:
            units = {'month': [11, start.month-1, until.month-1], 'hour': [23, start.hour, until.hour]}
            start_idx = int(np.round(((len(_pattern)-1) / units.get(limits)[0]) * (units.get(limits)[1]), 2))
            end_idx = int(np.round(((len(_pattern)-1) / units.get(limits)[0]) * (units.get(limits)[2]), 2))
            _pattern = _pattern[start_idx:end_idx+1]
        date_bins = pd.cut(date_range, bins=len(_pattern))
        index = DataBuilderTools._weighted_choice(_pattern, seed=seed)
        if index is None:
            return pd.NaT
        index_date = date_bins.categories[index]
        return pd.Timestamp(np.random.choice(pd.date_range(index_date.left, index_date.right, freq=freq)))

    @staticmethod
    def _weighted_choice(weights: list, seed: int=None):
        """ a probability weighting based on the values in the integer list

        :param weights: a list of integers representing a pattern of weighting
        :param seed: a seed value for the random function: default to None
        :return: an index of which weight was randomly chosen
        """
        if not isinstance(weights, list) or not all(isinstance(x, (int, float, list)) for x in weights):
            raise ValueError("The weighted pattern must be an list of integers")
        seed = DataBuilderTools._seed() if seed is None else seed
        np.random.seed(seed)
        rnd = np.random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    @staticmethod
    def _normailse_weights(weights: list, size: int = None, count: int = None, length: int=None):
        """normalises a complex weight pattern and returns the appropriate weight pattern
        based on the size and index.

        Example of a complex weight pattern might be:
            patten = [[1,2,3,1], 4, 6, [4,1]]

        :param weights: the weight pattern to normalise
        :param size: (optional) the total size of the selection
        :param count: (optional) the index in the selection to retrieve
        :param length: (optional) stretches or cuts the return pattern to the length
        :return: a list of weights
        """
        size = 1 if not isinstance(size, int) else size
        count = 0 if not isinstance(count, int) else count
        if not isinstance(weights, list):
            return [1]
        if count >= size:
            raise ValueError("counts can't be greater than or equal to size")
        pattern = []
        for i in weights:
            i = AbstractPropertiesManager.list_formatter(i)[:size]
            pattern.append(i)
        rtn_weights = []
        for p in pattern:
            if size == 1:
                index = 0
            else:
                index = int(np.round(((len(p) - 1) / (size - 1)) * count, 0))
            rtn_weights.append(p[index])
        if length is None:
            return rtn_weights
        if length <= len(rtn_weights):
            return rtn_weights[:length]
        rtn_pattern = []
        for i in range(length):
            index = int(np.round(((len(rtn_weights) - 1) / (length - 1)) * i, 2))
            rtn_pattern.append(rtn_weights[index])
        return rtn_pattern

    @staticmethod
    def _set_quantity(selection, quantity, seed=None):
        """Returns the quantity percent of good values in selection with the rest fill"""
        if quantity == 1:
            return selection
        seed = DataBuilderTools._seed() if seed is None else seed
        np.random.seed(seed)
        random.seed(seed)

        def replace_fill():
            """Used to run through all the possible fill options for the list type"""
            if isinstance(selection[i], float):
                selection[i] = np.nan
            elif isinstance(selection[i], str):
                selection[i] = ''
            else:
                selection[i] = None
            return

        if len(selection) < 100:
            for i in range(len(selection)):
                if np.random.random() > quantity:
                    replace_fill()
        else:
            sample_count = int(round(len(selection) * (1 - quantity), 0))
            indices = random.sample(list(range(len(selection))), sample_count)
            for i in indices:
                replace_fill()
        return selection

    @staticmethod
    def _mode_choice(values: list):
        """selects one or more of the most common occuring items in the list give. Acts like mode"""
        choice_list = []
        if values is None or len(values) == 0:
            return choice_list
        clean_list = [x for x in values if x is not None and
                      str(x) != 'nan' and str(x) is not 'None' and len(str(x)) != 0]
        if len(clean_list) == 0:
            return choice_list
        count_dict = Counter(clean_list)
        max_value = max(count_dict.values())
        for k, v in count_dict.items():
            if v == max_value:
                choice_list.append(k)
        return choice_list

    @staticmethod
    def _quantity(quantity: [float, int]) -> float:
        """normalises quantity to a percentate float between 0 and 1.0"""
        if not isinstance(quantity, (int, float)) or not 0 <= quantity <= 100:
            return 1.0
        if quantity > 1:
            return round(quantity/100, 2)
        return float(quantity)

    @staticmethod
    def _seed():
        return int(time.time() * np.random.random())

    @staticmethod
    def _next_seed(seed: int, default: int):
        seed += 1
        if seed > 2 ** 31:
            seed = DataBuilderTools._seed() if isinstance(default, int) else default
        np.random.seed(seed)
        return seed
