import inspect
import random
import re
import string
import threading
import warnings
from collections import Counter
from copy import deepcopy
from typing import Any, List
from matplotlib import dates as mdates
from pandas.tseries.offsets import Week
from ds_foundation.intent.abstract_intent import AbstractIntentModel
from ds_foundation.properties.abstract_properties import AbstractPropertyManager
from ds_foundation.handlers.abstract_handlers import ConnectorContract, HandlerFactory
from ds_behavioral.sample.sample_data import *

__author__ = 'Darryl Oatridge'


class SyntheticIntentModel(AbstractIntentModel):

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=True,
                 intent_next_available: bool=False):
        """initialisation of the Intent class. The 'intent_param_exclude' is used to exclude commonly used method
         parameters from being included in the intent contract, this is particularly useful if passing a canonical, or
         non relevant parameters to an intent method pattern. Any named parameter in the intent_param_exclude list
         will not be included in the recorded intent contract for that method

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param intent_next_available: (optional) if the default level should be set to next available level or zero
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        intent_param_exclude = ['df']
        super().__init__(property_manager=property_manager, intent_param_exclude=intent_param_exclude,
                         default_save_intent=default_save_intent)
        # globals
        self._default_intent_level = -1 if isinstance(intent_next_available, bool) and intent_next_available else 0

    def get_number(self, from_value: [int, float], to_value: [int, float]=None, weight_pattern: list=None, offset: int=None,
                   precision: int=None, currency: str=None, bounded_weighting: bool=True, at_most: int=None,
                   dominant_values: [float, list]=None, dominant_percent: float=None, dominance_weighting: list=None,
                   size: int = None, quantity: float=None, seed: int=None, save_intent: bool=True, intent_level: [int, str]=None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param offset: an offset multiplier, if None then assume 1
        :param currency: a currency symbol to prefix the value with. returns string with commas
        :param bounded_weighting: if the weighting pattern should have a soft or hard boundary constraint
        :param at_most: the most times a selection should be chosen
        :param dominant_values: a value or list of values with dominant_percent. if used MUST provide a dominant_percent
        :param dominant_percent: a value between 0 and 1 representing the dominant_percent of the dominant value(s)
        :param dominance_weighting: a weighting of the dominant values
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        # intent persist options
        if not isinstance(save_intent, bool):
            save_intent = self._default_save_intent
        if save_intent:
            intent_level = intent_level if isinstance(intent_level, (int, str)) else 0
            _intent_method = inspect.currentframe().f_code.co_name
            self._set_intend_signature(self._intent_builder(method=_intent_method, params=locals()), level=intent_level,
                                       save_intent=save_intent)
        # intent code
        (from_value, to_value) = (0, from_value) if not isinstance(to_value, (float, int)) else (from_value, to_value)
        at_most = 0 if not isinstance(at_most, int) else at_most
        if at_most > 0 and (at_most * (to_value-from_value)) < size:
            raise ValueError("When using 'at_most', the selectable values must be greater than the size. selectable "
                             "value count is '{}',size requested is '{}'".format(at_most * (to_value-from_value), size))
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        offset = 1 if offset is None else offset
        dominant_percent = 0 if not isinstance(dominant_percent, (int, float)) else dominant_percent
        dominant_percent = dominant_percent / 100 if 1 < dominant_percent <= 100 else dominant_percent
        _seed = self._seed() if seed is None else seed
        _limit = 10000
        precision = 3 if not isinstance(precision, int) else precision
        if precision == 0:
            from_value = int(round(from_value, 0))
            to_value = int(round(to_value, 0))
        is_int = True if isinstance(to_value, int) and isinstance(from_value, int) else False
        if is_int:
            precision = 0
        dominant_list = []
        if isinstance(dominant_values, (int, float, list)):
            sample_count = int(round(size * dominant_percent, 1)) if size > 1 else 0
            dominance_weighting = [1] if not isinstance(dominance_weighting, list) else dominance_weighting
            if sample_count > 0:
                if isinstance(dominant_values, list):
                    dominant_list = self.get_category(selection=dominant_values,
                                                                  weight_pattern=dominance_weighting,
                                                                  size=sample_count, bounded_weighting=True)
                else:
                    dominant_list = [dominant_values] * sample_count
            size -= sample_count
        if weight_pattern is not None:
            counter = [0] * len(weight_pattern)
            if bounded_weighting:
                unit = size/sum(weight_pattern)
                for i in range(len(weight_pattern)):
                    counter[i] = int(round(weight_pattern[i] * unit, 0))
                    if 0 < at_most < counter[i]:
                        counter[i] = at_most
                    if counter[i] == 0 and weight_pattern[i] > 0:
                        if counter[self._weighted_choice(weight_pattern)] == i:
                            counter[i] = 1
            else:
                for _ in range(size):
                    counter[self._weighted_choice(weight_pattern)] += 1
                for i in range(len(counter)):
                    if 0 < at_most < counter[i]:
                        counter[i] = at_most
            while sum(counter) != size:
                if at_most > 0:
                    for index in range(len(counter)):
                        if counter[index] >= at_most:
                            counter[index] = at_most
                            weight_pattern[index] = 0
                if sum(counter) < size:
                    counter[self._weighted_choice(weight_pattern)] += 1
                else:
                    weight_idx = self._weighted_choice(weight_pattern)
                    if counter[weight_idx] > 0:
                        counter[weight_idx] -= 1

        else:
            counter = [size]
        _seed = self._next_seed(_seed, seed)
        rtn_list = []
        if is_int:
            value_bins = []
            ref = from_value
            counter_len = len(counter)
            select_len = to_value - from_value
            for index in range(1, counter_len):
                position = int(round(select_len / counter_len * index, 1)) + from_value
                value_bins.append((ref, position))
                ref = position
            value_bins.append((ref, to_value))
            for index in range(counter_len):
                low, high = value_bins[index]
                if low >= high:
                    rtn_list += [low] * counter[index]
                elif at_most > 0:
                    choice = []
                    for _ in range(at_most):
                        choice += list(range(low, high))
                    np.random.shuffle(choice)
                    rtn_list += [int(np.round(value, precision)) for value in choice[:counter[index]]]
                else:
                    _remaining = counter[index]
                    while _remaining > 0:
                        _size = _limit if _remaining > _limit else _remaining
                        rtn_list += np.random.randint(low=low, high=high, size=_size).tolist()
                        _remaining -= _limit
        else:
            value_bins = pd.interval_range(start=from_value, end=to_value, periods=len(counter), closed='both')
            for index in range(len(counter)):
                low = value_bins[index].left
                high = value_bins[index].right
                if low >= high:
                    rtn_list += [low] * counter[index]
                elif at_most > 0:
                    choice = []
                    for _ in range(at_most):
                        choice += list(range(low, high))
                    np.random.shuffle(choice)
                    rtn_list += [np.round(value, precision) for value in choice[:counter[index]]]
                else:
                    _remaining = counter[index]
                    while _remaining > 0:
                        _size = _limit if _remaining > _limit else _remaining
                        rtn_list += np.round((np.random.random(size=_size)*(high-low)+low), precision).tolist()
                        _remaining -= _limit
        if isinstance(currency, str):
            rtn_list = ['{}{:0,.{}f}'.format(currency, value, precision) for value in rtn_list]
        if offset != 1:
            rtn_list = [value*offset for value in rtn_list]
        # add in the dominant values
        rtn_list = rtn_list + dominant_list
        np.random.shuffle(rtn_list)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_category(self, selection: list, weight_pattern: list=None, quantity: float=None, size: int=None,
                     bounded_weighting: bool=None, at_most: int=None, seed: int=None, save_intent: bool=True, intent_level: [int, str]=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.

        :param selection: a list of items to select from
        :param weight_pattern: a weighting pattern that does not have to add to 1
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: an optional size of the return. default to 1
        :param at_most: the most times a selection should be chosen
        :param bounded_weighting: if the weighting pattern should have a soft or hard boundary (default False)
        :param seed: a seed value for the random function: default to None
        :return: an item or list of items chosen from the list
        """
         # intent persist options
        if not isinstance(save_intent, bool):
            save_intent = self._default_save_intent
        if save_intent:
            intent_level = intent_level if isinstance(intent_level, (int, str)) else 0
            _intent_method = inspect.currentframe().f_code.co_name
            self._set_intend_signature(self._intent_builder(method=_intent_method, params=locals()), level=intent_level,
                                       save_intent=save_intent)
        # intent code
        if not isinstance(selection, list) or len(selection) == 0:
            return [None]*size
        bounded_weighting = False if not isinstance(bounded_weighting, bool) else bounded_weighting
        _seed = self._seed() if seed is None else seed
        quantity = self._quantity(quantity)
        select_index = self.get_number(len(selection), weight_pattern=weight_pattern, at_most=at_most,
                                                   size=size, bounded_weighting=bounded_weighting, quantity=1,
                                                   seed=seed)
        rtn_list = [selection[i] for i in select_index]
        return list(self._set_quantity(rtn_list, quantity=quantity, seed=_seed))

    def get_datetime(self, start: Any, until: Any, weight_pattern: list=None, at_most: int=None,  date_format: str=None,
                     as_num: bool=False, ignore_time: bool=False, size: int=None, quantity: float=None, seed: int=None,
                     day_first: bool=False, year_first: bool=False, save_intent: bool=True, intent_level: [int, str]=None) -> list:
        """ returns a random date between two date and times. weighted patterns can be applied to the overall date
        range, the year, month, day-of-week, hours and minutes to create a fully customised random set of dates.
        Note: If no patterns are set this will return a linearly random number between the range boundaries.
              Also if no patterns are set and a default date is given, that default date will be returnd each time

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param quantity: the quantity of values that are not null. Number between 0 and 1
        :param weight_pattern: (optional) A pattern across the whole date range.
        :param at_most: the most times a selection should be chosen
        :param ignore_time: ignore time elements and only select from Year, Month, Day elements. Default is False
        :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
        :param as_num: returns a list of Matplotlib date values as a float. Default is False
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
        # intent persist options
        if not isinstance(save_intent, bool):
            save_intent = self._default_save_intent
        if save_intent:
            intent_level = intent_level if isinstance(intent_level, (int, str)) else 0
            _intent_method = inspect.currentframe().f_code.co_name
            self._set_intend_signature(self._intent_builder(method=_intent_method, params=locals()), level=intent_level,
                                       save_intent=save_intent)
        # intent code
        as_num = False if not isinstance(as_num, bool) else as_num
        ignore_time = False if not isinstance(ignore_time, bool) else ignore_time
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        _dt_start = self._convert_date2value(start, day_first=day_first, year_first=year_first)[0]
        _dt_until = self._convert_date2value(until, day_first=day_first, year_first=year_first)[0]
        precision = 15
        if ignore_time:
            _dt_start = int(_dt_start)
            _dt_until = int(_dt_until)
            precision = 0
        rtn_list = self.get_number(from_value=_dt_start, to_value=_dt_until, weight_pattern=weight_pattern,
                                               at_most=at_most, precision=precision, size=size, seed=seed)
        if not as_num:
            rtn_list = mdates.num2date(rtn_list)
            if isinstance(date_format, str):
                rtn_list = pd.Series(rtn_list).dt.strftime(date_format).tolist()
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_intervals(self, intervals: list, weight_pattern: list=None, precision: int=None, currency: str=None,
                      size: int=None, quantity: float=None, seed: int=None, save_intent: bool=True, intent_level: [int, str]=None) -> list:
        """ returns a number based on a list selection of tuple(lower, upper) interval

        :param intervals: a list of unique tuple pairs representing the interval lower and upper boundaries
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param currency: a currency symbol to prefix the value with. returns string with commas
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
         # intent persist options
        if not isinstance(save_intent, bool):
            save_intent = self._default_save_intent
        if save_intent:
            intent_level = intent_level if isinstance(intent_level, (int, str)) else 0
            _intent_method = inspect.currentframe().f_code.co_name
            self._set_intend_signature(self._intent_builder(method=_intent_method, params=locals()), level=intent_level,
                                       save_intent=save_intent)
        # intent code
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        if not isinstance(precision, int):
            precision = 0 if all(isinstance(v[0], int) and isinstance(v[1], int) for v in intervals) else 3
        _seed = self._seed() if seed is None else seed
        if not all(isinstance(value, tuple) for value in intervals):
            raise ValueError("The intervals list but be a list of tuples")
        interval_list = self.get_category(selection=intervals, weight_pattern=weight_pattern, size=size,
                                                      seed=_seed)
        interval_counts = pd.Series(interval_list).value_counts()
        rtn_list = []
        for index in interval_counts.index:
            size = interval_counts[index]
            if size == 0:
                continue
            if len(index) == 2:
                (lower, upper) = index
                if index == 0:
                    closed = 'both'
                else:
                    closed = 'right'
            else:
                (lower, upper, closed) = index
            margin = 10**(((-1)*precision)-1)
            if str.lower(closed) == 'neither':
                lower += margin
                upper -= margin
            elif str.lower(closed) == 'left':
                upper -= margin
            elif str.lower(closed) == 'right':
                lower += margin
            rtn_list = rtn_list + self.get_number(lower, upper, precision=precision, currency=currency,
                                                              size=size, seed=_seed)
        np.random.seed(_seed)
        np.random.shuffle(rtn_list)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_distribution(self, method: str=None, offset: float=None, precision: int=None, size: int=None,
                         quantity: float=None, seed: int=None, save_intent: bool=True, intent_level: [int, str]=None, **kwargs) -> list:
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
        # intent persist options
        if not isinstance(save_intent, bool):
            save_intent = self._default_save_intent
        if save_intent:
            intent_level = intent_level if isinstance(intent_level, (int, str)) else 0
            _intent_method = inspect.currentframe().f_code.co_name
            self._set_intend_signature(self._intent_builder(method=_intent_method, params=locals()), level=intent_level,
                                       save_intent=save_intent)
        # intent code
        offset = 1 if offset is None or not isinstance(offset, (float, int)) else offset
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed

        method = 'normal' if method is None else method
        precision = 3 if precision is None else precision
        func = "np.random.{}(**{})".format(method, kwargs)
        rtn_list = []
        for _ in range(size):
            _seed = self._next_seed(_seed, seed)
            rtn_list.append(round(eval(func) * offset, precision))
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    """
        PRIVATE METHODS SECTION
    """
    @staticmethod
    def _filter_headers(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                        exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None) -> list:
        """ returns a list of headers based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes. Default is False
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :return: a filtered list of headers

        :raise: TypeError if any of the types are not as expected
        """
        if drop is None or not isinstance(drop, bool):
            drop = False
        if exclude is None or not isinstance(exclude, bool):
            exclude = False
        if re_ignore_case is None or not isinstance(re_ignore_case, bool):
            re_ignore_case = False

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The first function attribute must be a pandas 'DataFrame'")
        _headers = SyntheticIntentModel.list_formatter(headers)
        dtype = SyntheticIntentModel.list_formatter(dtype)
        regex = SyntheticIntentModel.list_formatter(regex)
        _obj_cols = df.columns
        _rtn_cols = set()
        unmodified = True

        if _headers is not None:
            _rtn_cols = set(_obj_cols).difference(_headers) if drop else set(_obj_cols).intersection(_headers)
            unmodified = False

        if regex is not None and regex:
            re_ignore_case = re.I if re_ignore_case else 0
            _regex_cols = list()
            for exp in regex:
                _regex_cols += [s for s in _obj_cols if re.search(exp, s, re_ignore_case)]
            _rtn_cols = _rtn_cols.union(set(_regex_cols))
            unmodified = False

        if unmodified:
            _rtn_cols = set(_obj_cols)

        if dtype is not None and len(dtype) > 0:
            _df_selected = df.loc[:, _rtn_cols]
            _rtn_cols = (_df_selected.select_dtypes(exclude=dtype) if exclude
                         else _df_selected.select_dtypes(include=dtype)).columns

        return [c for c in _rtn_cols]

    @staticmethod
    def _filter_columns(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                        inplace=False) -> [dict, pd.DataFrame]:
        """ Returns a subset of columns based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return:
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = SyntheticIntentModel._filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                    regex=regex, re_ignore_case=re_ignore_case)
        return df.loc[:, obj_cols]

    @staticmethod
    def _convert_date2value(dates: Any, day_first: bool = True, year_first: bool = False):
        values = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        return mdates.date2num(pd.Series(values)).tolist()



    @staticmethod
    def _convert_value2date(values: Any, date_format: str = None):
        dates = []
        for date in mdates.num2date(values):
            date = pd.Timestamp(date)
            if isinstance(date_format, str):
                date = date.strftime(date_format)
            dates.append(date)
        return dates

    def _date_choice(self, start: pd.Timestamp, until: pd.Timestamp, weight_pattern: list, limits: str=None, seed: int=None):
        """ Utility method to choose a random date between two dates based on a pattern.

        :param start: the start boundary
        :param until: the boundary to go up to
        :param weight_pattern: The weight pattern to apply to the range selection
        :param limits: (optional) time units that have pattern limits
        :param seed: (optional) a seed value for the random function: default to None
        :return: a choice from the range
        """
        seed = self._seed() if seed is None else seed
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
        index = self._weighted_choice(_pattern, seed=seed)
        if index is None:
            return pd.NaT
        index_date = date_bins.categories[index]
        return pd.Timestamp(np.random.choice(pd.date_range(index_date.left, index_date.right, freq=freq)))

    def _weighted_choice(self, weights: list, seed: int=None):
        """ a probability weighting based on the values in the integer list

        :param weights: a list of integers representing a pattern of weighting
        :param seed: a seed value for the random function: default to None
        :return: an index of which weight was randomly chosen
        """
        if not isinstance(weights, list) or not all(isinstance(x, (int, float, list)) for x in weights):
            raise ValueError("The weighted pattern must be an list of integers")
        seed = self._seed() if seed is None else seed
        np.random.seed(seed)
        rnd = np.random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    def _normailse_weights(self, weights: list, size: int = None, count: int = None, length: int=None):
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
            i = self.list_formatter(i)[:size]
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

    def _set_quantity(self, selection, quantity, seed=None):
        """Returns the quantity percent of good values in selection with the rest fill"""
        if quantity == 1:
            return selection
        seed = self._seed() if seed is None else seed
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

    def _next_seed(self, seed: int, default: int):
        seed += 1
        if seed > 2 ** 31:
            seed = self._seed() if isinstance(default, int) else default
        np.random.seed(seed)
        return seed

    @staticmethod
    def list_formatter(value) -> [List[str], list, None]:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series into a list"""
        if isinstance(value, (int, float, str, pd.Timestamp)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, dict):
            return list(value.items())
        return None
