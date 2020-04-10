import inspect
import itertools
import random
import re
import string
import warnings
from copy import deepcopy
from typing import Any
from matplotlib import dates as mdates
from pandas.tseries.offsets import Week

from aistac.intent.abstract_intent import AbstractIntentModel
from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_behavioral.components.commons import Commons, DataAnalytics
from ds_behavioral.sample.sample_data import *

__author__ = 'Darryl Oatridge'


class SyntheticIntentModel(AbstractIntentModel):

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=None,
                 default_intent_level: bool=None, order_next_available: bool=None, default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'A'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['canonical', 'size']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, size: int, columns: [str, list]=None, **kwargs) -> pd.DataFrame:
        """Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        :param size: the size of the outcome data set
        :param columns: (optional) a single or list of intent_level to run, if list, run in order given
        :param kwargs: additional parameters to pass beyond the contracted parameters
        :return: a pandas dataframe
        """
        df = pd.DataFrame()
        # test if there is any intent to run
        if self._pm.has_intent():
            # size
            size = size if isinstance(size, int) else 1000
            # get the list of levels to run
            if isinstance(columns, (str, list)):
                column_names = self._pm.list_formatter(columns)
            else:
                # put all the intent in order of model, get, correlate, associate
                _model = []
                _get = []
                _correlate = []
                _associate = []
                _remove = []
                for column in self._pm.get_intent().keys():
                    for order in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column), {}):
                        for method in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column, order), {}).keys():
                            if str(method).startswith('model_'):
                                _model.append(column)
                            elif str(method).startswith('get_'):
                                _get.append(column)
                            elif str(method).startswith('correlate_'):
                                _correlate.append(column)
                            elif str(method).startswith('associate_'):
                                _associate.append(column)
                            elif str(method).startswith('remove_'):
                                _remove.append(column)
                column_names = Commons.unique_list(_model + _get + _correlate + _associate + _remove)
            for column in column_names:
                level_key = self._pm.join(self._pm.KEY.intent_key, column)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        if method in self.__dir__():
                            result = []
                            params.update(params.pop('kwargs', {}))
                            if isinstance(kwargs, dict):
                                params.update(kwargs)
                            if str(method).startswith('get_'):
                                result = eval(f"self.{method}(size=size, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('correlate_'):
                                corr_column = params.get('header', None)
                                if corr_column is None:
                                    raise NameError(f"The column name '{column}' can't be found for correlation. "
                                                    f"check the referenced column is generated before correlation")
                                result = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('associate_'):
                                result = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('model_'):
                                result = eval(f"self.{method}(size=size, save_intent=False, **params)",
                                              globals(), locals())
                                result = pd.DataFrame(result)
                                df = pd.concat([df, result], axis=1, sort=False, copy=False)
                                continue
                            elif str(method).startswith('remove_'):
                                df = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                          globals(), locals())
                                continue
                            if len(result) != size:
                                raise IndexError(f"The index size of '{column}' is '{len(result)}', should be {size}")
                            df[column] = result
        return df

    def get_number(self, range_value: [int, float]=None, to_value: [int, float]=None, weight_pattern: list=None,
                   offset: int=None, precision: int=None, currency: str=None, bounded_weighting: bool=None,
                   at_most: int=None, dominant_values: [float, list]=None, dominant_percent: float=None,
                   dominance_weighting: list=None, size: int=None, quantity: float=None, seed: int=None,
                   save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                   replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param range_value: range value to_value if to_value is used else from 0 to value if to_value is None
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
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(range_value, (int, float)) and not isinstance(to_value, (int, float)):
            raise ValueError(f"either a 'to_value' or a 'from_value' and to_value' must be provided as a parameter")
        if not isinstance(range_value, (float, int)):
            range_value = 0
        if not isinstance(to_value, (float, int)):
            (range_value, to_value) = (0, range_value)
        bounded_weighting = bounded_weighting if isinstance(bounded_weighting, bool) else True
        at_most = 0 if not isinstance(at_most, int) else at_most
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        offset = 1 if offset is None else offset
        dominant_percent = 0 if not isinstance(dominant_percent, (int, float)) else dominant_percent
        dominant_percent = dominant_percent / 100 if 1 < dominant_percent <= 100 else dominant_percent
        _seed = self._seed() if seed is None else seed
        _limit = 10000
        precision = 3 if not isinstance(precision, int) else precision
        if precision == 0:
            range_value = int(round(range_value, 0))
            to_value = int(round(to_value, 0))
        is_int = True if (isinstance(to_value, int) and isinstance(range_value, int)) else False
        if is_int:
            precision = 0
        dominant_list = []
        if isinstance(dominant_values, (int, float, list)):
            sample_count = int(round(size * dominant_percent, 1)) if size > 1 else 0
            dominance_weighting = [1] if not isinstance(dominance_weighting, list) else dominance_weighting
            if sample_count > 0:
                if isinstance(dominant_values, list):
                    dominant_list = self.get_category(selection=dominant_values, weight_pattern=dominance_weighting,
                                                      size=sample_count, bounded_weighting=True, save_intent=False)
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
            ref = range_value
            counter_len = len(counter)
            select_len = to_value - range_value
            for index in range(1, counter_len):
                position = int(round(select_len / counter_len * index, 1)) + range_value
                value_bins.append((ref, position))
                ref = position
            value_bins.append((ref, to_value))
            for index in range(counter_len):
                low, high = value_bins[index]
                # check our range is not also the dominants
                if high - low <= 1:
                    rtn_list += [low] * counter[index]
                elif at_most > 0:
                    check_list = [[] for index in range(at_most)]
                    total_len = 0
                    chk_count = 0
                    while total_len < counter[index]:
                        chk_count += 1
                        total_len = 0
                        for _group in range(at_most):
                            check_list[_group] += np.random.randint(low, high, counter[index]).tolist()
                            check_list[_group] = [i for i in check_list[_group] if i not in
                                                  Commons.list_formatter(dominant_values)] + [high]
                            total_len += len(check_list[_group])
                    if chk_count > 25:
                        raise RecursionError(f"The selectable value pool is too small to get "
                                             f"at most '{at_most}' of size {size}")

                    rtn_list += list(itertools.chain.from_iterable(check_list))[:counter[index]]
                    np.random.shuffle(rtn_list)
                else:
                    _remaining = counter[index]
                    while _remaining > 0:
                        _size = _limit if _remaining > _limit else _remaining
                        choice = list(np.random.randint(low=low, high=high, size=_size))
                        choice = [i for i in choice if i not in Commons.list_formatter(dominant_values)]
                        rtn_list += choice
                        _remaining -= len(choice)
        else:
            value_bins = pd.interval_range(start=range_value, end=to_value, periods=len(counter), closed='both')
            for index in range(len(counter)):
                low = value_bins[index].left
                high = value_bins[index].right
                if low >= high:
                    rtn_list += [low] * counter[index]
                elif at_most > 0:
                    check_list = [[] for index in range(at_most)]
                    total_len = 0
                    chk_count = 0
                    while total_len < counter[index]:
                        chk_count += 1
                        total_len = 0
                        for _group in range(at_most):
                            check_list[_group] += np.round(
                                (np.random.random(size=int(counter[index] / at_most)+1)*(high-low)+low),
                                precision).tolist()
                            check_list[_group] = list(set(check_list[_group]))
                            check_list[_group] = [i for i in check_list[_group] if i not in
                                                  Commons.list_formatter(dominant_values) + [high]]
                            total_len += len(check_list[_group])
                        if chk_count > 25:
                            raise RecursionError(f"The selectable value pool is too small to get "
                                                 f"at most '{at_most}' of size {size}")
                    rtn_list += list(itertools.chain.from_iterable(check_list))[:counter[index]]
                    np.random.shuffle(rtn_list)
                else:
                    _remaining = counter[index]
                    while _remaining > 0:
                        _size = _limit if _remaining > _limit else _remaining
                        choice = np.round((np.random.random(size=_size)*(high-low)+low), precision).tolist()
                        choice = [i for i in choice if i not in Commons.list_formatter(dominant_values) + [high]]
                        rtn_list += choice
                        _remaining -= len(choice)
        if isinstance(currency, str):
            rtn_list = ['{}{:0,.{}f}'.format(currency, value, precision) for value in rtn_list]
        if offset != 1:
            rtn_list = [value*offset for value in rtn_list]
        # add in the dominant values
        rtn_list += dominant_list
        np.random.shuffle(rtn_list)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_category(self, selection: list, weight_pattern: list=None, quantity: float=None, size: int=None,
                     bounded_weighting: bool=None, at_most: int=None, seed: int=None, save_intent: bool=None,
                     column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.

        :param selection: a list of items to select from
        :param weight_pattern: a weighting pattern that does not have to add to 1
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: an optional size of the return. default to 1
        :param at_most: the most times a selection should be chosen
        :param bounded_weighting: if the weighting pattern should have a soft or hard boundary (default False)
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an item or list of items chosen from the list
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(selection, list) or len(selection) == 0:
            return [None]*size
        bounded_weighting = bounded_weighting if isinstance(bounded_weighting, bool) else False
        _seed = self._seed() if seed is None else seed
        quantity = self._quantity(quantity)
        select_index = self.get_number(len(selection), weight_pattern=weight_pattern, at_most=at_most, size=size,
                                       bounded_weighting=bounded_weighting, quantity=1, seed=seed, save_intent=False)
        rtn_list = [selection[i] for i in select_index]
        return list(self._set_quantity(rtn_list, quantity=quantity, seed=_seed))

    def get_datetime(self, start: Any, until: Any, weight_pattern: list=None, at_most: int=None, date_format: str=None,
                     as_num: bool=None, ignore_time: bool=None, size: int=None, quantity: float=None, seed: int=None,
                     day_first: bool=None, year_first: bool=None, save_intent: bool=None, column_name: [int, str]=None,
                     intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a random date between two date and/or times. weighted patterns can be applied to the overall date
        range.
        if a signed 'int' type is passed to the start and/or until dates, the inferred date will be the current date
        time with the integer being the offset from the current date time in 'days'.

        Note: If no patterns are set this will return a linearly random number between the range boundaries.

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
        :param until: up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
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
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a date or size of dates in the format given.
         """
        # pre check
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        as_num = False if not isinstance(as_num, bool) else as_num
        ignore_time = False if not isinstance(ignore_time, bool) else ignore_time
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        if isinstance(start, int):
            start = (pd.Timestamp.now() + pd.Timedelta(days=start))
        if isinstance(until, int):
            until = (pd.Timestamp.now() + pd.Timedelta(days=until))
        _dt_start = self._convert_date2value(start, day_first=day_first, year_first=year_first)[0]
        _dt_until = self._convert_date2value(until, day_first=day_first, year_first=year_first)[0]
        precision = 15
        if ignore_time:
            _dt_start = int(_dt_start)
            _dt_until = int(_dt_until)
            precision = 0
        rtn_list = self.get_number(range_value=_dt_start, to_value=_dt_until, weight_pattern=weight_pattern,
                                   at_most=at_most, precision=precision, size=size, seed=seed, save_intent=False)
        if not as_num:
            rtn_list = mdates.num2date(rtn_list)
            if isinstance(date_format, str):
                rtn_list = pd.Series(rtn_list).dt.strftime(date_format).tolist()
            else:
                rtn_list = pd.Series(rtn_list).dt.tz_convert(None).to_list()
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_datetime_pattern(self, start: Any, until: Any, default: Any=None, ordered: bool=None,
                             date_pattern: list=None, year_pattern: list=None, month_pattern: list=None,
                             weekday_pattern: list=None, hour_pattern: list=None, minute_pattern: list=None,
                             quantity: float=None, date_format: str=None, size: int=None, seed: int=None,
                             day_first: bool = True, year_first: bool = False, save_intent: bool=None,
                             column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> list:
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
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a date or size of dates in the format given.
         """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        ordered = False if not isinstance(ordered, bool) else ordered
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
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
            _seed = self._next_seed(_seed, seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message='Discarding nonzero nanoseconds in conversion')
                _min_date = (pd.Timestamp.min + pd.DateOffset(years=1)).replace(month=1, day=1, hour=0, minute=0,
                                                                                second=0, microsecond=0, nanosecond=0)
                _max_date = (pd.Timestamp.max + pd.DateOffset(years=-1)).replace(month=12, day=31, hour=23, minute=59,
                                                                                 second=59, microsecond=0, nanosecond=0)
                # reset the starting base
            _dt_default = _dt_base
            if not isinstance(_dt_default, pd.Timestamp):
                _dt_default = np.random.random() * (_dt_until - _dt_start) + pd.to_timedelta(_dt_start)
            # ### date ###
            if date_pattern is not None:
                _dp_start = self._convert_date2value(_dt_start)[0]
                _dp_until = self._convert_date2value(_dt_until)[0]
                value = self.get_number(_dp_start, _dp_until, weight_pattern=date_pattern, seed=_seed,
                                        save_intent=False)
                _dt_default = self._convert_value2date(value)[0]
            # ### years ###
            rand_year = _dt_default.year
            if year_pattern is not None:
                rand_select = self._date_choice(_dt_start, _dt_until, year_pattern, seed=_seed)
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
                rand_select = self._date_choice(month_start, month_end, month_pattern, limits='month', seed=_seed)
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
                _weekday = self._weighted_choice(weekday_pattern, seed=_seed)
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
                rand_select = self._date_choice(hour_start, hour_end, hour_pattern, limits='hour', seed=seed)
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
                rand_select = self._date_choice(minute_start, minute_end, minute_pattern, seed=seed)
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
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_intervals(self, intervals: list, weight_pattern: list=None, precision: int=None, currency: str=None,
                      dominant_values: [float, list]=None, dominant_percent: float=None, dominance_weighting: list=None,
                      size: int=None, quantity: float=None, seed: int=None, save_intent: bool=None,
                      column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                      remove_duplicates: bool=None) -> list:
        """ returns a number based on a list selection of tuple(lower, upper) interval

        :param intervals: a list of unique tuple pairs representing the interval lower and upper boundaries
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param dominant_values: a value or list of values with dominant_percent. if used MUST provide a dominant_percent
        :param dominant_percent: a value between 0 and 1 representing the dominant_percent of the dominant value(s)
        :param dominance_weighting: a weighting of the dominant values
        :param currency: a currency symbol to prefix the value with. returns string with commas
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        if not isinstance(precision, int):
            precision = 0 if all(isinstance(v[0], int) and isinstance(v[1], int) for v in intervals) else 3
        _seed = self._seed() if seed is None else seed
        if not all(isinstance(value, tuple) for value in intervals):
            raise ValueError("The intervals list must be a list of tuples")
        interval_list = self.get_category(selection=intervals, weight_pattern=weight_pattern, size=size, seed=_seed,
                                          save_intent=False)
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
            if precision == 0:
                margin = 1
            else:
                margin = 10**(((-1)*precision)-1)
            if str.lower(closed) == 'neither':
                lower += margin
                upper -= margin
            elif str.lower(closed) == 'right':
                lower += margin
            elif str.lower(closed) == 'both':
                upper += margin
            rtn_list = rtn_list + self.get_number(lower, upper, precision=precision, currency=currency, size=size,
                                                  dominant_values=dominant_values,
                                                  dominance_weighting=dominance_weighting,
                                                  dominant_percent=dominant_percent, seed=_seed, save_intent=False)
        np.random.seed(_seed)
        np.random.shuffle(rtn_list)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_distribution(self, method: str=None, offset: float=None, precision: int=None, size: int=None,
                         quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                         intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                         **kwargs) -> list:
        """returns a number based the distribution type. Supports Normal, Beta and

        :param method: any method name under np.random. Default is 'normal'
        :param offset: a value to offset the number by. n * offset
        :param precision: the precision of the returned number
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param kwargs: the parameters of the method
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
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

    def get_string_pattern(self, pattern: str, choices: dict=None, quantity: [float, int]=None, size: int=None,
                           choice_only: bool=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                           intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
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
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a string based on the pattern
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        choice_only = False if choice_only is None or not isinstance(choice_only, bool) else choice_only
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
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
                        "The key '{}' must contain a 'list' of replacements opotions. '{}' found".format(k, type(v)))

        rtn_list = []
        _seed = self._next_seed(_seed, seed)
        for c in list(pattern):
            if c in choices.keys():
                result = np.random.choice(choices[c], size=size)
            elif not choice_only:
                result = [c]*size
            else:
                continue
            rtn_list = [i + j for i, j in zip(rtn_list, result)] if len(rtn_list) > 0 else result
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_from(self, connector_name: str, column_header: str, weight_pattern: list=None, selection_size: int=None,
                 sample_size: int=None, size: int=None, at_most: bool=None, shuffled: bool=None, quantity: float=None,
                 seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                 replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a random list of values where the selection of those values is taken from the values passed.

        :param connector_name: a connector_name for a connector to a data source
        :param column_header: the name of the column header to correlate
        :param weight_pattern: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, normally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param at_most: (optional) the most times a selection should be chosen
        :param shuffled: (optional) if the selection should be shuffled before selection. Default is true
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param size: (optional) size of the return. default to 1
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return:
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        _seed = self._seed() if seed is None else seed
        if not self._pm.has_connector(connector_name=connector_name):
            raise ValueError(f"The connector name '{connector_name}' is not in the connectors catalog")
        handler = self._pm.get_connector_handler(connector_name)
        canonical = handler.load_canonical()
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        self._pm.set_modified(connector_name, handler.get_modified())
        if column_header not in canonical.columns:
            raise ValueError(f"The column '{column_header}' not found in the data from connector '{connector_name}'")
        _values = canonical[column_header].iloc[:sample_size]
        if isinstance(selection_size, bool) and shuffled:
            _values = _values.sample(frac=1).reset_index(drop=True)
        if isinstance(selection_size, int) and 0 < selection_size < _values.size:
            _values = _values.iloc[:selection_size]
        return self.get_category(selection=_values.tolist(), weight_pattern=weight_pattern, quantity=quantity,
                                 size=size, at_most=at_most, seed=_seed, save_intent=False)

    def get_profile_middle_name(self, size: int, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                                intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """local method to generate a middle initial """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        middle = self.get_category(selection=list("ABCDEFGHIJKLMNOPRSTW") + ['  '] * 4, size=int(size * 0.95),
                                   seed=seed, save_intent=False)
        choices = {'U': list("ABCDEFGHIJKLMNOPRSTW")}
        middle += self.get_string_pattern(pattern="U U", choices=choices, choice_only=False, size=size - len(middle),
                                          seed=seed, save_intent=False)
        return middle

    def get_profile_surname(self, size: int, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                            intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """local method to generate a middle initial """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        return self.get_category(selection=ProfileSample.surnames(), size=size, seed=seed, save_intent=False)

    def get_identifiers(self, from_value: int, to_value: int=None, size: int=None, prefix: str=None, suffix: str=None,
                        quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                        intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a list of unique identifiers randomly selected between the from_value and to_value

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param size: the size of the sample. Must be smaller than the range
        :param prefix: a prefix to the number . Default to nothing
        :param suffix: a suffix to the number. default to nothing
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a unique identifer randomly selected from the range
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        (from_value, to_value) = (0, from_value) if not isinstance(to_value, (float, int)) else (from_value, to_value)
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        rtn_list = []
        for i in self.get_number(range_value=from_value, to_value=to_value, at_most=1, size=size, precision=0,
                                 seed=seed, save_intent=False):
            rtn_list.append("{}{}{}".format(prefix, i, suffix))
        return self._set_quantity(rtn_list, quantity=quantity, seed=seed)

    def get_tagged_pattern(self, pattern: [str, list], tags: dict, weight_pattern: list=None, size: int=None,
                           quantity: [float, int]=None, seed: int=None, save_intent: bool=None,
                           column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                           remove_duplicates: bool=None) -> list:
        """ Returns the pattern with the tags substituted by tag choice
            example ta dictionary:
                { '<slogan>': {'action': '', 'kwargs': {}},
                  '<phone>': {'action': '', 'kwargs': {}}
                }
            where action is a self method name and kwargs are the arguments to pass
            for sample data use get_custom

        :param pattern: a string or list of strings to apply the ta substitution too
        :param tags: a dictionary of tas and actions
        :param weight_pattern: a weighting pattern that does not have to add to 1
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: an optional size of the return. default to 1
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of patterns with tas replaced
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        pattern = self._pm.list_formatter(pattern)
        if not isinstance(tags, dict):
            raise ValueError("The 'tags' parameter must be a dictionary")
        class_methods = self.__dir__

        rtn_list = []
        for _ in range(size):
            _seed = self._next_seed(_seed, seed)
            choice = self.get_category(pattern, weight_pattern=weight_pattern, seed=_seed, size=1, save_intent=False)[0]
            for tag, action in tags.items():
                method = action.get('action')
                if method in class_methods:
                    kwargs = action.get('kwargs')
                    result = eval("self.{}(**{})".format(method, kwargs))[0]
                else:
                    result = method
                choice = re.sub(tag, str(result), str(choice))
            rtn_list.append(choice)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_custom(self, code_str: str, quantity: float=None, size: int=None, seed: int=None, save_intent: bool=None,
                   column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                   remove_duplicates: bool=None, **kwargs) -> list:
        """returns a number based on the random func. The code should generate a value per line
        example:
            code_str = 'round(np.random.normal(loc=loc, scale=scale), 3)'
            fbt.get_custom(code_str, loc=0.4, scale=0.1)

        :param code_str: an evaluable code as a string
        :param quantity: (optional) a number between 0 and 1 representing data that isn't null
        :param size: (optional) the size of the sample
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random value based on function called
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed

        rtn_list = []
        for _ in range(size):
            _seed = self._next_seed(_seed, seed)
            local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
            rtn_list.append(eval(code_str, globals(), local_kwargs))
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def remove_columns(self, canonical: pd.DataFrame, headers: list, save_intent: bool=None,
                       column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                       remove_duplicates: bool=None):
        """local method to generate a forename with dominance

        :param canonical: a DataFrame that contains a column to correlate
        :param headers: the headers of the columns to remove
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        return canonical.drop(headers, axis=1)

    def model_us_zip(self, rename_columns: dict, size: int=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                     intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ builds a model of distributed Zipcode, City and State with weighting towards the more populated zipcodes

        :param rename_columns: (optional) rename the columns 'City', 'Zipcode', 'State'
        :param size: (optional) the size. should be greater than or equal to the analysis sample for best results.
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a dictionary
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        _seed = self._seed() if seed is None else seed
        size = 1 if size is None else size
        df = MappedSample.us_zipcode_primary(cleaned=True)
        df_high = df.where(df['EstimatedPopulation'] > 20000).dropna()
        df_low = df.where(df['EstimatedPopulation'] <= 20000).dropna()
        df = df.sort_values(by='EstimatedPopulation', ascending=False)
        low_size = int(0.001 * size)
        high_size = size - low_size
        idx = self.get_number(df_high.shape[0], weight_pattern=[10, 7, 6, 5, 4, 3, 2, 0.9] + [0.6]*50 + [0.3]*50,
                              seed=seed, size=high_size, save_intent=False)
        df_rtn = df_high.iloc[idx]
        idx = self.get_number(df_low.shape[0], size=low_size, seed=seed, save_intent=False)
        df_rtn = df_rtn.append(df_low.iloc[idx])
        df_rtn = Commons.filter_columns(df_rtn, headers=['City', 'Zipcode', 'State'])
        df_rtn['Zipcode'] = df['Zipcode'].round(0).astype(int)
        if isinstance(rename_columns, dict):
            df_rtn = df_rtn.rename(columns=rename_columns)
        return df_rtn.sample(frac=1).reset_index(drop=True).to_dict(orient='list')

    def model_analysis(self, analytics_model: dict, size: int=None, seed: int=None, save_intent: bool=None,
                       column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                       remove_duplicates: bool=None) -> dict:
        """ builds a set of columns based on an analysis dictionary of weighting (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param analytics_model: the analytics model from discovery-tranistion-ds discovery model train
        :param size: (optional) the size. should be greater than or equal to the analysis sample for best results.
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a dictionary
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)

        def get_level(analysis: dict, sample_size: int):
            for name, values in analysis.items():
                if row_dict.get(name) is None:
                    row_dict[name] = list()
                _analysis = DataAnalytics(label=name, analysis=values.get('analysis', {}))
                if str(_analysis.intent.dtype).startswith('cat'):
                    row_dict[name] += self.get_category(selection=_analysis.intent.selection,
                                                        weight_pattern=_analysis.patterns.weight_pattern,
                                                        quantity=1-_analysis.stats.nulls_percent, seed=seed,
                                                        size=sample_size, save_intent=save_intent, column_name=name,
                                                        replace_intent=replace_intent,
                                                        remove_duplicates=remove_duplicates)
                if str(_analysis.intent.dtype).startswith('num'):
                    row_dict[name] += self.get_intervals(intervals=_analysis.intent.selection,
                                                         weight_pattern=_analysis.patterns.weight_pattern,
                                                         dominant_values=_analysis.patterns.dominant_values,
                                                         dominant_percent=_analysis.patterns.dominant_percent,
                                                         dominance_weighting=_analysis.patterns.dominance_weighting,
                                                         precision=_analysis.intent.precision,
                                                         quantity=1 - _analysis.stats.nulls_percent,
                                                         seed=seed, size=sample_size, save_intent=save_intent,
                                                         column_name=name, replace_intent=replace_intent,
                                                         remove_duplicates=remove_duplicates)
                if str(_analysis.intent.dtype).startswith('date'):
                    row_dict[name] += self.get_datetime(start=_analysis.intent.lower, until=_analysis.intent.upper,
                                                        weight_pattern=_analysis.patterns.weight_pattern,
                                                        date_format=_analysis.intent.data_format,
                                                        day_first=_analysis.intent.day_first,
                                                        year_first=_analysis.intent.year_first,
                                                        quantity=1 - _analysis.stats.nulls_percent,
                                                        seed=seed, size=sample_size, save_intent=save_intent,
                                                        column_name=name, replace_intent=replace_intent,
                                                        remove_duplicates=remove_duplicates)
                unit = sample_size / sum(_analysis.patterns.weight_pattern)
                if values.get('sub_category'):
                    section_map = _analysis.weight_map
                    for i in section_map.index:
                        section_size = int(round(_analysis.weight_map.loc[i] * unit, 0))+1
                        next_item = values.get('sub_category').get(i)
                        get_level(next_item, section_size)
            return

        row_dict = dict()
        size = 1 if not isinstance(size, int) else size
        get_level(analytics_model, sample_size=size)
        for key in row_dict.keys():
            row_dict[key] = row_dict[key][:size]
        return row_dict

    def associate_canonical(self, canonical: Any, associations: list, actions: dict, default_value: Any=None,
                            default_header: str=None, day_first: bool=None, quantity: float=None, seed: int=None,
                            save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                            replace_intent: bool=None, remove_duplicates: bool=None):
        """ Associates a set of criteria of an input values to a set of actions
            The association dictionary takes the form of a set of dictionaries in a list with each item in the list
            representing an index key for the action dictionary. Each dictionary are an associated relationship.
            In this example for the first index the associated values should be in header1  and within a date range
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
            in the association, that list index is used as reference to the action.
                {0: {'action': '', 'kwargs' : {}},
                 1: {...}}
            you can also use the action to specify a specific value:
                {0: {'action': ''},
                 1: {'action': ''}}

        :param canonical: the data set to map against, this can be a str, int, float, list, Series or DataFrame
        :param associations: a list of categories (can also contain lists for multiple references.
        :param actions: the correlated set of categories that should map to the index
        :param default_header: (optional) if no association, the default column header to take the value from.
                    if None then the default_value is taken.
                    Note for non-DataFrame datasets the default header is '_default'
        :param default_value: (optional) if no default header then this value is taken if no association
        :param day_first: (optional) if expected type is date, indicates if the day is first. Default to true
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        _seed = self._seed() if seed is None else seed

        if not isinstance(canonical, (str, int, float, list, pd.Series, pd.DataFrame)):
            raise TypeError("The parameter values is not an accepted type")
        if not isinstance(associations, (list, dict)):
            raise TypeError("The parameter reference must be a list or dict")
        _dataset = canonical
        _associations = associations
        if isinstance(_dataset, (str, int, float)):
            _dataset = self._pm.list_formatter(_dataset)
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
        class_methods = self.__dir__

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
                    chk_value = self._pm.list_formatter(lookup.get('value'))
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
                        if v.get('_header') not in canonical.columns:
                            raise ValueError("Dataset header '{}' does not exist: see action: {} -> key: {}".format(
                                v.get('_header'), action_idx, k))
                        kwargs[k] = canonical[v.get('_header')].iloc[index]
                result = eval("self.{}(**{})".format(method, kwargs).replace('nan', 'None'))
                if isinstance(result, list):
                    if len(result) > 0:
                        rtn_list.append(result.pop())
                    else:
                        rtn_list.append(None)
                else:
                    rtn_list.append(result)
            else:
                if isinstance(method, dict):
                    if method.get('_header') not in canonical.columns:
                        raise ValueError("Dataset header '{}' does not exist: see action: {} -> key: action".format(
                            method.get('_header'), action_idx))
                    rtn_list.append(canonical[method.get('_header')].iloc[index])
                else:
                    rtn_list.append(method)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def associate_custom(self, canonical: pd.DataFrame, code_str: str, use_exec: bool=None, save_intent: bool=None,
                         column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None, **kwargs):
        """ enacts an action on a dataFrame, returning the output of the action or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your action, it is internally referenced
        as it's parameter name 'canonical'.

        :param canonical: a pd.DataFrame used in the action
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list or pandas.DataFrame
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        use_exec = use_exec if isinstance(use_exec, bool) else False
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if result is None:
            return canonical
        return result

    def correlate_from_columns(self, canonical: pd.DataFrame, headers: list, sep: str=None, seed: int=None,
                               save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                               replace_intent: bool=None, remove_duplicates: bool=None):
        """local method to generate a forename with dominance

        :param canonical: a DataFrame that contains a column to correlate
        :param headers: an ordered list of columns to join
        :param sep: (optional) a separator value between each of the value
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        sep = sep if isinstance(sep, str) else ''
        headers = Commons.list_formatter(headers)
        return canonical[headers].applymap(str).agg(sep.join, axis=1).to_list()

    def correlate_forename_to_gender(self, canonical: pd.DataFrame, header: str, categories: list, seed: int=None,
                                     save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                                     replace_intent: bool=None, remove_duplicates: bool=None):
        """correlate a forename to a gender column so as to matche the gender to an appropriate first name

        :param canonical: a DataFrame that contains a column to correlate
        :param header: the header in the DataFrame to correlate
        :param categories: a list of length two with the male then female category label to correlate e.g. ['M', 'F']
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # validation
        if not isinstance(categories, list) or not len(categories) == 2:
            raise ValueError(f"The categories must list the Male and Female label to correlate, e.g. ['M', 'F']")
        if not isinstance(canonical, pd.DataFrame):
            raise ValueError(f"The canonical must be a pandas DataFrame")
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        s_values = canonical[header].copy()
        _seed = seed if isinstance(seed, int) else self._seed()
        m_index = s_values[s_values == categories[0]].index
        f_index = s_values[s_values == categories[1]].index
        m_names = self.get_category(selection=ProfileSample.male_names(seed=_seed), size=m_index.size, seed=_seed,
                                    save_intent=False)
        f_names = self.get_category(selection=ProfileSample.female_names(seed=_seed), size=f_index.size, seed=_seed,
                                    save_intent=False)
        result = pd.Series(data=[np.nan] * s_values.size)
        result.loc[m_index] = m_names
        result.loc[f_index] = f_names
        return result.to_list()

    def correlate_numbers(self, canonical: pd.DataFrame, header: str, spread: float=None, offset: float=None,
                          weighting_pattern: list=None, multiply_offset: bool=None, precision: int=None,
                          fill_nulls: bool=None, quantity: float=None, seed: int=None, keep_zero: bool=None,
                          min_value: [int, float]=None, max_value: [int, float]=None, save_intent: bool=None,
                          column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ returns a number that correlates to the value given. The spread is based on a normal distribution
        with the value being the mean and the spread its standard deviation from that mean

        :param canonical: a DataFrame that contains a column to correlate
        :param header: the header in the DataFrame to correlate
        :param spread: (optional) the random spread or deviation from the value. defaults to 0
        :param offset: (optional) how far from the value to offset. defaults to zero
        :param weighting_pattern: a weighting pattern with the pattern mid point the mid point of the spread
        :param multiply_offset: (optional) if true then the offset is multiplied else added
        :param precision: (optional) how many decimal places. default to 3
        :param fill_nulls: (optional) if True then fills nulls with the most common values
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) the random seed. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
        :param min_value: a minimum value not to go below
        :param max_value: a max value not to go above
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(canonical, pd.DataFrame):
            raise ValueError(f"The canonical must be a pandas DataFrame")
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        fill_nulls = fill_nulls if isinstance(fill_nulls, bool) else False
        keep_zero = keep_zero if isinstance(keep_zero, bool) else False
        precision = precision if isinstance(precision, int) else 3
        action = 'multiply' if isinstance(multiply_offset, bool) and multiply_offset else 'add'
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        if fill_nulls:
            s_values = s_values.fillna(np.random.choice(s_values.mode(dropna=True)))
        null_idx = s_values[s_values.isna()].index
        zero_idx = s_values.where(s_values == 0).dropna().index if keep_zero else []
        if isinstance(offset, (int, float)) and offset != 0:
            s_values = s_values.mul(offset) if action == 'multiply' else s_values.add(offset)
        if isinstance(spread, (int, float)) and spread != 0:
            sample = self.get_number(-abs(spread) / 2, abs(spread) / 2, weight_pattern=weighting_pattern,
                                     size=s_values.size, save_intent=False)
            s_values = s_values.add(sample)
        if isinstance(min_value, (int, float)):
            if min_value < s_values.max():
                min_idx = s_values.dropna().where(s_values < min_value).dropna().index
                s_values.iloc[min_idx] = min_value
            else:
                raise ValueError(f"The min value {min_value} is greater than the max result value {s_values.max()}")
        if isinstance(max_value, (int, float)):
            if max_value > s_values.min():
                max_idx = s_values.dropna().where(s_values > max_value).dropna().index
                s_values.iloc[max_idx] = max_value
            else:
                raise ValueError(f"The max value {max_value} is less than the min result value {s_values.min()}")
        # reset the zero values if any
        s_values.iloc[zero_idx] = 0
        s_values = s_values.round(precision)
        if precision == 0 and not s_values.isnull().any():
            s_values = s_values.astype(int)
        if null_idx.size > 0:
            s_values.iloc[null_idx] = np.nan
        return self._set_quantity(s_values.tolist(), quantity=quantity, seed=_seed)

    def correlate_categories(self, canonical: pd.DataFrame, header: str, correlations: list, actions: dict,
                             fill_nulls: bool=None, quantity: float=None, seed: int=None,
                             save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None, 
                             replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlation of a set of values to an action, the correlations must map to the dictionary index values.
        Note. to use the current value in the passed values as a parameter value pass an empty dict {} as the keys
        value. If you want the action value to be the current value of the passed value then again pass an empty dict
        action to be the current value
            simple correlation list:
                ['A', 'B', 'C'] # if values is 'A' then action is 0 and so on
            multiple choice correlation
                [['A','B'], 'C'] # if values is 'A' OR 'B' then action is 0 and so on
            actions dictionary where the method is a class method followed by its parameters
                {0: {'method': 'get_numbers', 'from_value': 0, to_value: 27}}
            you can also use the action to specify a specific value:
                {0: 'F', 1: {'method': 'get_numbers', 'from_value': 0, to_value: 27}}

        :param canonical: a DataFrame that contains a column to correlate
        :param header: the header in the DataFrame to correlate
        :param correlations: a list of categories (can also contain lists for multiple correlations.
        :param actions: the correlated set of categories that should map to the index
        :param fill_nulls: (optional) if True then fills nulls with the most common values
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(canonical, pd.DataFrame):
            raise ValueError(f"The canonical must be a pandas DataFrame")
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        fill_nulls = fill_nulls if isinstance(fill_nulls, bool) else False
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        actions = deepcopy(actions)
        correlations = deepcopy(correlations)
        if fill_nulls:
            s_values = s_values.fillna(np.random.choice(s_values.mode(dropna=True)))
        null_idx = s_values[s_values.isna()].index
        s_values.to_string()
        corr_list = []
        for corr in correlations:
            corr_list.append(self._pm.list_formatter(corr))
        class_methods = self.__dir__()
        for i in range(len(corr_list)):
            corr_idx = s_values[s_values.isin(corr_list[i])].index
            action = actions.get(i, -1)
            if action is -1:
                continue
            if isinstance(action, dict):
                method = action.pop('method', None)
                if method is None:
                    raise ValueError(f"The action key '{i}' dictionary has no 'method' key.")
                if method in class_methods:
                    params = actions.get(i, {})
                    if not isinstance(params, dict):
                        params = {}
                    params.update({'size': corr_idx.size})
                    data = eval(f"self.{method}(**params)", globals(), locals())
                    result = pd.Series(data=data, index=corr_idx)
                else:
                    raise ValueError(f"The 'method' key {method} is not a recognised intent method")
            else:
                result = pd.Series(data=(action * corr_idx.size), index=corr_idx)
            s_values.update(result)
        if null_idx.size > 0:
            s_values.iloc[null_idx] = np.nan
        return self._set_quantity(s_values.tolist(), quantity=quantity, seed=_seed)

    def correlate_dates(self, canonical: pd.DataFrame, header: str, offset: [int, dict]=None, spread: int=None,
                        spread_units: str=None, spread_pattern: list=None, date_format: str=None,
                        min_date: str=None, max_date: str=None, fill_nulls: bool=None, day_first: bool=None,
                        year_first: bool=None, quantity: float=None, seed: int=None, save_intent: bool=None,
                        column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None):
        """ correlates dates to an existing date or list of dates.

        :param canonical: a DataFrame that contains a column to correlate
        :param header: the header in the DataFrame to correlate
        :param offset: (optional) and offset to the date. if int then assumed a 'days' offset
                int or dictionary associated with pd. eg {'days': 1}
        :param spread: (optional) the random spread or deviation in days
        :param spread_units: (optional) the units of the spread, Options: 'W', 'D', 'h', 'm', 's'. default 'D'
        :param spread_pattern: (optional) a weighting pattern with the pattern mid point the mid point of the spread
        :param min_date: (optional)a minimum date not to go below
        :param max_date: (optional)a max date not to go above
        :param fill_nulls: (optional) if no date values should remain untouched or filled based on the list mode date
        :param day_first: (optional) if the dates given are day first firmat. Default to True
        :param year_first: (optional) if the dates given are year first. Default to False
        :param date_format: (optional) the format of the output
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal size to that given
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(canonical, pd.DataFrame):
            raise ValueError(f"The canonical must be a pandas DataFrame")
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        values = canonical[header].copy()
        if values.empty:
            return list()

        def _clean(control):
            _unit_type = ['years', 'months', 'weeks', 'days', 'leapdays', 'hours', 'minutes', 'seconds']
            _params = {}
            if isinstance(control, int):
                control = {'days': control}
            if isinstance(control, dict):
                for k, v in control.items():
                    if k not in _unit_type:
                        raise ValueError(f"The key '{k}' in 'offset', is not a recognised unit type for pd.DateOffset")
            return control

        quantity = self._quantity(quantity)
        _seed = self._seed() if seed is None else seed
        fill_nulls = False if fill_nulls is None or not isinstance(fill_nulls, bool) else fill_nulls
        offset = _clean(offset) if isinstance(offset, (dict, int)) else None
        units_allowed = ['W', 'D', 'h', 'm', 's']
        spread_units = spread_units if isinstance(spread_units, str) and spread_units in units_allowed else 'D'
        spread = pd.Timedelta(value=spread, unit=spread_units) if isinstance(spread, int) else None
        # set minimum date
        _min_date = pd.to_datetime(min_date, errors='coerce', infer_datetime_format=True, utc=True)
        if _min_date is None or _min_date is pd.NaT:
            _min_date = pd.to_datetime(pd.Timestamp.min, utc=True)
        # set max date
        _max_date = pd.to_datetime(max_date, errors='coerce', infer_datetime_format=True, utc=True)
        if _max_date is None or _max_date is pd.NaT:
            _max_date = pd.to_datetime(pd.Timestamp.max, utc=True)
        if _min_date >= _max_date:
            raise ValueError(f"the min_date {min_date} must be less than max_date {max_date}")
        # convert values into datetime
        s_values = pd.Series(pd.to_datetime(values.copy(), errors='coerce', infer_datetime_format=True,
                                            dayfirst=day_first, yearfirst=year_first, utc=True))
        if spread is not None:
            if spread_units in ['W', 'D']:
                value = spread.days
                zip_units = 'D'
            else:
                value = int(spread.to_timedelta64().astype(int)/1000000000)
                zip_units = 's'
            zip_spread = self.get_number(-abs(value) / 2, (abs(value+1) / 2), weight_pattern=spread_pattern,
                                         precision=0, size=s_values.size, save_intent=False)
            zipped_dt = list(zip(zip_spread, [zip_units]*s_values.size))
            s_values = s_values + np.array([pd.Timedelta(x, y).to_timedelta64() for x, y in zipped_dt])
        if fill_nulls:
            s_values = s_values.fillna(np.random.choice(s_values.mode(dropna=True)))
        null_idx = s_values[s_values.isna()].index
        if isinstance(offset, dict) and offset:
            s_values = s_values.add(pd.DateOffset(**offset))
        if _min_date > pd.to_datetime(pd.Timestamp.min, utc=True):
            if _min_date > s_values.min():
                min_idx = s_values.dropna().where(s_values < _min_date).dropna().index
                s_values.iloc[min_idx] = _min_date
            else:
                raise ValueError(f"The min value {min_date} is greater than the max result value {s_values.max()}")
        if _max_date < pd.to_datetime(pd.Timestamp.max, utc=True):
            if _max_date < s_values.max():
                max_idx = s_values.dropna().where(s_values > _max_date).dropna().index
                s_values.iloc[max_idx] = _max_date
            else:
                raise ValueError(f"The max value {max_date} is less than the min result value {s_values.min()}")
        if isinstance(date_format, str):
            s_values = s_values.dt.strftime(date_format)
        else:
            s_values = s_values.dt.tz_convert(None)
        if null_idx.size > 0:
            s_values.iloc[null_idx].apply(lambda x: np.nan)
        return self._set_quantity(s_values.tolist(), quantity=quantity, seed=_seed)

    """
        PRIVATE METHODS SECTION
    """

    def _set_intend_signature(self, intent_params: dict, column_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        if save_intent or (not isinstance(save_intent, bool) and self._default_save_intent):
            if not isinstance(column_name, (str, int)) or not column_name:
                raise ValueError(f"if the intent is to be saved then a column name must be provided")
        super()._set_intend_signature(intent_params=intent_params, intent_level=column_name, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return

    @staticmethod
    def _convert_date2value(dates: Any, day_first: bool = True, year_first: bool = False):
        values = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        return mdates.date2num(pd.Series(values)).tolist()

    @staticmethod
    def _convert_value2date(values: Any, date_format: str=None):
        dates = []
        for date in mdates.num2date(values):
            date = pd.Timestamp(date)
            if isinstance(date_format, str):
                date = date.strftime(date_format)
            dates.append(date)
        return dates

    def _date_choice(self, start, until, weight_pattern: list, limits: str=None,
                     seed: int=None):
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

    def _normailse_weights(self, weights: list, size: int=None, count: int=None, length: int=None):
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
            i = self._pm.list_formatter(i)[:size]
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
    def _quantity(quantity: [float, int]) -> float:
        """normalises quantity to a percentate float between 0 and 1.0"""
        if not isinstance(quantity, (int, float)) or not 0 <= quantity <= 100:
            return 1.0
        if quantity > 1:
            return round(quantity / 100, 2)
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
