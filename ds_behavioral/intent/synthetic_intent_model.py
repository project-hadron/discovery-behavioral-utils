import inspect
import random
import re
import string
import ast
import time
import numpy as np
import pandas as pd
from uuid import UUID, uuid1, uuid3, uuid4, uuid5
from copy import deepcopy
from typing import Any
from matplotlib import dates as mdates
from scipy import stats

from aistac.intent.abstract_intent import AbstractIntentModel
from aistac.handlers.abstract_handlers import HandlerFactory
from ds_behavioral.components.commons import SyntheticCommons, DataAnalytics
from ds_behavioral.managers.synthetic_property_manager import SyntheticPropertyManager
from ds_behavioral.sample.sample_data import MappedSample, Sample

__author__ = 'Darryl Oatridge'


class SyntheticIntentModel(AbstractIntentModel):

    def __init__(self, property_manager: SyntheticPropertyManager, default_save_intent: bool=None,
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

    def run_intent_pipeline(self, size: int, columns: [str, list]=None, seed: int=None) -> pd.DataFrame:
        """Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract. The whole run can be seeded though any parameterised seeding in the intent
        contracts will take precedence

        :param size: the size of the outcome data set
        :param columns: (optional) a single or list of intent_level to run, if list, run in order given
        :param seed: a seed value that will be applied across the run: default to None
        :return: a pandas dataframe
        """
        df = pd.DataFrame()
        # test if there is any intent to run
        if self._pm.has_intent():
            # size
            size = size if isinstance(size, int) else 10
            # get the list of levels to run
            if isinstance(columns, (str, list)):
                column_names = self._pm.list_formatter(columns)
            else:
                # put all the intent in order of model, get, correlate, associate
                _model = []
                _get = []
                _correlate = []
                _frame = []
                for column in self._pm.get_intent().keys():
                    for order in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column), {}):
                        for method in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column, order), {}).keys():
                            if str(method).startswith('get_'):
                                _get.append(column)
                            elif str(method).startswith('model_'):
                                _model.append(column)
                            elif str(method).startswith('correlate_'):
                                if column in _get:
                                    _get.remove(column)
                                _correlate.append(column)
                            elif str(method).startswith('frame_'):
                                if column in _get:
                                    _get.remove(column)
                                _frame.append(column)
                column_names = SyntheticCommons.unique_list(_get + _model + _correlate + _frame)
            for column in column_names:
                level_key = self._pm.join(self._pm.KEY.intent_key, column)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        if method in self.__dir__():
                            result = []
                            params.update(params.pop('kwargs', {}))
                            if isinstance(seed, int):
                                params.update({'seed': seed})
                            _ = params.pop('intent_creator', 'Unknown')
                            if str(method).startswith('get_'):
                                result = eval(f"self.{method}(size=size, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('correlate_'):
                                result = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('model_'):
                                df = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                          globals(), locals())
                                continue
                            elif str(method).startswith('frame_'):
                                df = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                          globals(), locals())
                                continue
                            if len(result) != size:
                                raise IndexError(f"The index size of '{column}' is '{len(result)}', should be {size}")
                            df[column] = result
        return df

    def get_number(self, from_value: [int, float]=None, to_value: [int, float]=None, relative_freq: list=None,
                   precision: int=None, ordered: str=None, at_most: int=None, size: int=None, quantity: float=None,
                   seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                   replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: (signed) integer to start from
        :param to_value: optional, (signed) integer the number sequence goes to but not include
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param ordered: order the data ascending 'asc' or descending 'dec', values accepted 'asc' or 'des'
        :param at_most: the most times a selection should be chosen
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
        if not isinstance(from_value, (int, float)) and not isinstance(to_value, (int, float)):
            raise ValueError(f"either a 'range_value' or a 'range_value' and 'to_value' must be provided")
        if not isinstance(from_value, (float, int)):
            from_value = 0
        if not isinstance(to_value, (float, int)):
            (from_value, to_value) = (0, from_value)
        if to_value <= from_value:
            raise ValueError("The number range must be a positive different, found to_value <= from_value")
        at_most = 0 if not isinstance(at_most, int) else at_most
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        precision = 3 if not isinstance(precision, int) else precision
        if precision == 0:
            from_value = int(round(from_value, 0))
            to_value = int(round(to_value, 0))
        is_int = True if (isinstance(to_value, int) and isinstance(from_value, int)) else False
        if is_int:
            precision = 0
        # build the distribution sizes
        if isinstance(relative_freq, list) and len(relative_freq) > 1:
            freq_dist_size = self._freq_dist_size(relative_freq=relative_freq, size=size, seed=_seed)
        else:
            freq_dist_size = [size]
        # generate the numbers
        rtn_list = []
        generator = np.random.default_rng(seed=_seed)
        dtype = int if is_int else float
        bins = np.linspace(from_value, to_value, len(freq_dist_size) + 1, dtype=dtype, endpoint=True)
        for idx in np.arange(1, len(bins)):
            low = bins[idx - 1]
            high = bins[idx]
            if low >= high:
                continue
            elif at_most > 0:
                sample = []
                for count in np.arange(at_most, dtype=dtype):
                    count_size = freq_dist_size[idx - 1] * generator.integers(2, 4, size=1)[0]
                    sample += list(set(np.linspace(bins[idx - 1], bins[idx], num=count_size, dtype=dtype,
                                                   endpoint=False)))
                if len(sample) < freq_dist_size[idx - 1]:
                    raise ValueError(f"The value range has insufficient samples to choose from when using at_most."
                                     f"Try increasing the range of values to sample.")
                rtn_list += list(generator.choice(sample, size=freq_dist_size[idx - 1], replace=False))
            else:
                if dtype == int:
                    rtn_list += generator.integers(low=low, high=high, size=freq_dist_size[idx - 1]).tolist()
                else:
                    choice = generator.random(size=freq_dist_size[idx - 1], dtype=float)
                    choice = np.round(choice * (high-low)+low, precision).tolist()
                    # make sure the precision
                    choice = [high - 10**(-precision) if x >= high else x for x in choice]
                    rtn_list += choice
        # order or shuffle the return list
        if isinstance(ordered, str) and ordered.lower() in ['asc', 'des']:
            rtn_list.sort(reverse=True if ordered.lower() == 'asc' else False)
        else:
            generator.shuffle(rtn_list)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_category(self, selection: list, relative_freq: list=None, quantity: float=None, size: int=None,
                     bounded_weighting: bool=None, at_most: int=None, seed: int=None, save_intent: bool=None,
                     column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.

        :param selection: a list of items to select from
        :param relative_freq: a weighting pattern that does not have to add to 1
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
        select_index = self.get_number(len(selection), relative_freq=relative_freq, at_most=at_most, size=size,
                                       quantity=1, seed=_seed, save_intent=False)
        rtn_list = [selection[i] for i in select_index]
        return list(self._set_quantity(rtn_list, quantity=quantity, seed=_seed))

    def get_datetime(self, start: Any, until: Any, relative_freq: list=None, at_most: int=None, ordered: str=None,
                     date_format: str=None, as_num: bool=None, ignore_time: bool=None, size: int=None,
                     quantity: float=None, seed: int=None, day_first: bool=None, year_first: bool=None,
                     save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                     replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a random date between two date and/or times. weighted patterns can be applied to the overall date
        range.
        if a signed 'int' type is passed to the start and/or until dates, the inferred date will be the current date
        time with the integer being the offset from the current date time in 'days'.

        Note: If no patterns are set this will return a linearly random number between the range boundaries.

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
        :param until: up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
        :param quantity: the quantity of values that are not null. Number between 0 and 1
        :param relative_freq: (optional) A pattern across the whole date range.
        :param at_most: the most times a selection should be chosen
        :param ordered: order the data ascending 'asc' or descending 'dec', values accepted 'asc' or 'des'
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
        rtn_list = self.get_number(from_value=_dt_start, to_value=_dt_until, relative_freq=relative_freq,
                                   at_most=at_most, ordered=ordered, precision=precision, size=size, seed=seed,
                                   save_intent=False)
        if not as_num:
            rtn_list = mdates.num2date(rtn_list)
            if isinstance(date_format, str):
                rtn_list = pd.Series(rtn_list).dt.strftime(date_format).to_list()
            else:
                rtn_list = pd.Series(rtn_list).dt.tz_convert(None).to_list()
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    # def get_datetime_pattern(self, start: Any, until: Any, default: Any=None, ordered: bool=None,
    #                          year_pattern: list=None, month_pattern: list=None, weekday_pattern: list=None,
    #                          hour_pattern: list=None, minute_pattern: list=None, quantity: float=None,
    #                          date_format: str=None, size: int=None, seed: int=None, day_first: bool=None,
    #                          year_first: bool=None, save_intent: bool=None, column_name: [int, str]=None,
    #                          intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
    #     """ returns a random date between two date and times. weighted patterns can be applied to the overall date
    #     range, the year, month, day-of-week, hours and minutes to create a fully customised random set of dates.
    #     Note: If no patterns are set this will return a linearly random number between the range boundaries.
    #           Also if no patterns are set and a default date is given, that default date will be returnd each time
    #
    #     :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
    #     :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
    #     :param default: (optional) a fixed starting date that patterns are applied too.
    #     :param ordered: (optional) if the return list should be date ordered. Default is True
    #     :param year_pattern: (optional) adjusts the year selection to this pattern
    #     :param month_pattern: (optional) adjusts the month selection to this pattern. Must be of length 12
    #     :param weekday_pattern: (optional) adjusts the weekday selection to this pattern. Must be of length 7
    #     :param hour_pattern: (optional) adjusts the hours selection to this pattern. must be of length 24
    #     :param minute_pattern: (optional) adjusts the minutes selection to this pattern
    #     :param quantity: the quantity of values that are not null. Number between 0 and 1
    #     :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
    #     :param size: the size of the sample to return. Default to 1
    #     :param seed: a seed value for the random function: default to None
    #     :param year_first: specifies if to parse with the year first
    #             If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
    #             If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
    #     :param day_first: specifies if to parse with the day first
    #             If True, parses dates with the day first, eg %d-%m-%Y.
    #             If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
    #     :param save_intent (optional) if the intent contract should be saved to the property manager
    #     :param column_name: (optional) the column name that groups intent to create a column
    #     :param intent_order: (optional) the order in which each intent should run.
    #                     If None: default's to -1
    #                     if -1: added to a level above any current instance of the intent section, level 0 if not found
    #                     if int: added to the level specified, overwriting any that already exist
    #     :param replace_intent: (optional) if the intent method exists at the level, or default level
    #                     True - replaces the current intent method with the new
    #                     False - leaves it untouched, disregarding the new intent
    #     :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
    #     :return: a date or size of dates in the format given.
    #      """
    #     # intent persist options
    #    self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
    #                                column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
    #                                remove_duplicates=remove_duplicates, save_intent=save_intent)
    #     # Code block for intent
    #     # TODO: All this data pattern could be replaced by correlation layering, needs investigating
    #     ordered = ordered if isinstance(ordered, bool) else True
    #     if start is None or until is None:
    #         raise ValueError("The start or until parameters cannot be of NoneType")
    #     quantity = self._quantity(quantity)
    #     size = size if isinstance(size, int) else 1
    #     _seed = seed if isinstance(seed, int) else self._seed()
    #     if default:
    #         date_values = [pd.to_datetime(default, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
    #                                       yearfirst=year_first)] * size
    #     else:
    #         date_values = self.get_datetime(start=start, until=until, date_format=date_format, day_first=day_first,
    #                                         year_first=year_first, seed=_seed, size=size, save_intent=False)
    #     date_values = pd.Series(date_values)
    #     # filter by year
    #     for _year in date_values.dt.year.unique():
    #         yr_idx = date_values.where(date_values.dt.year == _year).dropna().index

    # def get_datetime_pattern(self, start: Any, until: Any, default: Any = None, ordered: bool = None,
    #                          date_pattern: list = None, year_pattern: list = None, month_pattern: list = None,
    #                          weekday_pattern: list = None, hour_pattern: list = None, minute_pattern: list = None,
    #                          quantity: float = None, date_format: str = None, size: int = None, seed: int = None,
    #                          day_first: bool = True, year_first: bool = False, save_intent: bool = None,
    #                          column_name: [int, str] = None, intent_order: int = None, replace_intent: bool = None,
    #                          remove_duplicates: bool = None) -> list:
    #     """ returns a random date between two date and times. weighted patterns can be applied to the overall date
    #     range, the year, month, day-of-week, hours and minutes to create a fully customised random set of dates.
    #     Note: If no patterns are set this will return a linearly random number between the range boundaries.
    #           Also if no patterns are set and a default date is given, that default date will be returnd each time
    #
    #     :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
    #     :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
    #     :param default: (optional) a fixed starting date that patterns are applied too.
    #     :param ordered: (optional) if the return list should be date ordered
    #     :param date_pattern: (optional) A pattern across the whole date range.
    #             If set, is the primary pattern with each subsequent pattern overriding this result
    #             If no other pattern is set, this will return a random date based on this pattern
    #     :param year_pattern: (optional) adjusts the year selection to this pattern
    #     :param month_pattern: (optional) adjusts the month selection to this pattern. Must be of length 12
    #     :param weekday_pattern: (optional) adjusts the weekday selection to this pattern. Must be of length 7
    #     :param hour_pattern: (optional) adjusts the hours selection to this pattern. must be of length 24
    #     :param minute_pattern: (optional) adjusts the minutes selection to this pattern
    #     :param quantity: the quantity of values that are not null. Number between 0 and 1
    #     :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
    #     :param size: the size of the sample to return. Default to 1
    #     :param seed: a seed value for the random function: default to None
    #     :param year_first: specifies if to parse with the year first
    #             If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
    #             If both day_first and year_first are True, year_first is preceded (same as dateutil).
    #     :param day_first: specifies if to parse with the day first
    #             If True, parses dates with the day first, eg %d-%m-%Y.
    #             If False default to the a preferred preference, normally %m-%d-%Y (but not strict)
    #     :param save_intent (optional) if the intent contract should be saved to the property manager
    #     :param column_name: (optional) the column name that groups intent to create a column
    #     :param intent_order: (optional) the order in which each intent should run.
    #                     If None: default's to -1
    #                     if -1: added to a level above any current instance of the intent section, level 0 if not found
    #                     if int: added to the level specified, overwriting any that already exist
    #     :param replace_intent: (optional) if the intent method exists at the level, or default level
    #                     True - replaces the current intent method with the new
    #                     False - leaves it untouched, disregarding the new intent
    #     :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
    #     :return: a date or size of dates in the format given.
    #      """
    #     # intent persist options
    #    self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
    #                                column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
    #                                remove_duplicates=remove_duplicates, save_intent=save_intent)
    #     # Code block for intent
    #     ordered = False if not isinstance(ordered, bool) else ordered
    #     if start is None or until is None:
    #         raise ValueError("The start or until parameters cannot be of NoneType")
    #     quantity = self._quantity(quantity)
    #     size = 1 if size is None else size
    #     _seed = self._seed() if seed is None else seed
    #     _dt_start = pd.to_datetime(start, errors='coerce', infer_datetime_format=True,
    #                                dayfirst=day_first, yearfirst=year_first)
    #     _dt_until = pd.to_datetime(until, errors='coerce', infer_datetime_format=True,
    #                                dayfirst=day_first, yearfirst=year_first)
    #     _dt_base = pd.to_datetime(default, errors='coerce', infer_datetime_format=True,
    #                               dayfirst=day_first, yearfirst=year_first)
    #     if _dt_start is pd.NaT or _dt_until is pd.NaT:
    #         raise ValueError("The start or until parameters cannot be converted to a timestamp")
    #
    #     # ### Apply the patterns if any ###
    #     rtn_dates = []
    #     for _ in range(size):
    #         _seed = self._next_seed(_seed, seed)
    #         with warnings.catch_warnings():
    #             warnings.filterwarnings("ignore", message='Discarding nonzero nanoseconds in conversion')
    #             _min_date = (pd.Timestamp.min + pd.DateOffset(years=1)).replace(month=1, day=1, hour=0, minute=0,
    #                                                                             second=0, microsecond=0, nanosecond=0)
    #             _max_date = (pd.Timestamp.max + pd.DateOffset(years=-1)).replace(month=12, day=31, hour=23, minute=59,
    #                                                                            second=59, microsecond=0, nanosecond=0)
    #             # reset the starting base
    #         _dt_default = _dt_base
    #         if not isinstance(_dt_default, pd.Timestamp):
    #             _dt_default = np.random.random() * (_dt_until - _dt_start) + pd.to_timedelta(_dt_start)
    #         # ### date ###
    #         if date_pattern is not None:
    #             _dp_start = self._convert_date2value(_dt_start)[0]
    #             _dp_until = self._convert_date2value(_dt_until)[0]
    #             value = self.get_number(_dp_start, _dp_until, relative_freq=date_pattern, seed=_seed,
    #                                     save_intent=False)
    #             _dt_default = self._convert_value2date(value)[0]
    #         # ### years ###
    #         rand_year = _dt_default.year
    #         if year_pattern is not None:
    #             rand_select = self._date_choice(_dt_start, _dt_until, year_pattern, seed=_seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_year = rand_select.year
    #         _max_date = _max_date.replace(year=rand_year)
    #         _min_date = _min_date.replace(year=rand_year)
    #         _dt_default = _dt_default.replace(year=rand_year)
    #         # ### months ###
    #         rand_month = _dt_default.month
    #         rand_day = _dt_default.day
    #         if month_pattern is not None:
    #             month_start = _dt_start if _dt_start.year == _min_date.year else _min_date
    #             month_end = _dt_until if _dt_until.year == _max_date.year else _max_date
    #             rand_select = self._date_choice(month_start, month_end, month_pattern, limits='month', seed=_seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_month = rand_select.month
    #             rand_day = _dt_default.day if _dt_default.day <= rand_select.daysinmonth else rand_select.daysinmonth
    #         _max_date = _max_date.replace(month=rand_month, day=rand_day)
    #         _min_date = _min_date.replace(month=rand_month, day=rand_day)
    #         _dt_default = _dt_default.replace(month=rand_month, day=rand_day)
    #         # ### weekday ###
    #         if weekday_pattern is not None:
    #             if not len(weekday_pattern) == 7:
    #                 raise ValueError("The weekday_pattern mut be a list of size 7 with index 0 as Monday")
    #             _weekday = self._weighted_choice(weekday_pattern, seed=_seed)
    #             if _weekday != _min_date.dayofweek:
    #                 if _dt_start <= (_dt_default + Week(weekday=_weekday)) <= _dt_until:
    #                     rand_day = (_dt_default + Week(weekday=_weekday)).day
    #                     rand_month = (_dt_default + Week(weekday=_weekday)).month
    #                 elif _dt_start <= (_dt_default - Week(weekday=_weekday)) <= _dt_until:
    #                     rand_day = (_dt_default - Week(weekday=_weekday)).day
    #                     rand_month = (_dt_default - Week(weekday=_weekday)).month
    #                 else:
    #                     rtn_dates.append(pd.NaT)
    #                     continue
    #         _max_date = _max_date.replace(month=rand_month, day=rand_day)
    #         _min_date = _min_date.replace(month=rand_month, day=rand_day)
    #         _dt_default = _dt_default.replace(month=rand_month, day=rand_day)
    #         # ### hour ###
    #         rand_hour = _dt_default.hour
    #         if hour_pattern is not None:
    #             hour_start = _dt_start if _min_date.strftime('%d%m%Y') == _dt_start.strftime('%d%m%Y') else _min_date
    #             hour_end = _dt_until if _max_date.strftime('%d%m%Y') == _dt_until.strftime('%d%m%Y') else _max_date
    #             rand_select = self._date_choice(hour_start, hour_end, hour_pattern, limits='hour', seed=seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_hour = rand_select.hour
    #         _max_date = _max_date.replace(hour=rand_hour)
    #         _min_date = _min_date.replace(hour=rand_hour)
    #         _dt_default = _dt_default.replace(hour=rand_hour)
    #         # ### minutes ###
    #         rand_minute = _dt_default.minute
    #         if minute_pattern is not None:
    #             minute_start = _dt_start \
    #                 if _min_date.strftime('%d%m%Y%H') == _dt_start.strftime('%d%m%Y%H') else _min_date
    #             minute_end = _dt_until \
    #                 if _max_date.strftime('%d%m%Y%H') == _dt_until.strftime('%d%m%Y%H') else _max_date
    #             rand_select = self._date_choice(minute_start, minute_end, minute_pattern, seed=seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_minute = rand_select.minute
    #         _max_date = _max_date.replace(minute=rand_minute)
    #         _min_date = _min_date.replace(minute=rand_minute)
    #         _dt_default = _dt_default.replace(minute=rand_minute)
    #         # ### get the date ###
    #         _dt_default = _dt_default.replace(second=np.random.randint(60))
    #         if isinstance(_dt_default, pd.Timestamp):
    #             _dt_default = _dt_default.tz_localize(None)
    #         rtn_dates.append(_dt_default)
    #     if ordered:
    #         rtn_dates = sorted(rtn_dates)
    #     rtn_list = []
    #     if isinstance(date_format, str):
    #         for d in rtn_dates:
    #             if isinstance(d, pd.Timestamp):
    #                 rtn_list.append(d.strftime(date_format))
    #             else:
    #                 rtn_list.append(str(d))
    #     else:
    #         rtn_list = rtn_dates
    #     return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_intervals(self, intervals: list, relative_freq: list=None, precision: int=None, size: int=None,
                      quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                      intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a number based on a list selection of tuple(lower, upper) interval

        :param intervals: a list of unique tuple pairs representing the interval lower and upper boundaries
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
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
        interval_list = self.get_category(selection=intervals, relative_freq=relative_freq, size=size, seed=_seed,
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
            rtn_list = rtn_list + self.get_number(lower, upper, precision=precision, size=size, seed=_seed,
                                                  save_intent=False)
        np.random.default_rng(seed=_seed).shuffle(rtn_list)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_dist_normal(self, mean: float, std: float, size: int=None, quantity: float=None, seed: int=None,
                        save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A normal (Gaussian) continuous random distribution.

        :param mean: The mean (“centre”) of the distribution.
        :param std: The standard deviation (jitter or “width”) of the distribution. Must be >= 0
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
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
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.normal(loc=mean, scale=std, size=size))
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_dist_binomial(self, trials: int, probability: float, size: int=None, quantity: float=None, seed: int=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A binomial discrete random distribution. The Binomial Distribution represents the number of
           successes and failures in n independent Bernoulli trials for some given value of n

        :param trials: the number of trials to attempt, must be >= 0.
        :param probability: the probability distribution, >= 0 and <=1.
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
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
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.binomial(n=trials, p=probability, size=size))
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_dist_poisson(self, interval: float, size: int=None, quantity: float=None, seed: int=None,
                         save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A Poisson discrete random distribution

        :param interval: Expectation of interval, must be >= 0.
        :param size: the size of the sample.
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
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.poisson(lam=interval, size=size))
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_dist_bernoulli(self, probability: float, size: int=None, quantity: float=None, seed: int=None,
                           save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A Bernoulli discrete random distribution using scipy

        :param probability: the probability occurrence
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
        _seed = self._seed() if seed is None else seed
        rtn_list = list(stats.bernoulli.rvs(p=probability, size=size, random_state=_seed))
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_dist_bounded_normal(self, mean: float, std: float, lower: float, upper: float, precision: int=None,
                                size: int=None, quantity: float=None, seed: int=None, save_intent: bool=None,
                                column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                                remove_duplicates: bool=None) -> list:
        """A bounded normal continuous random distribution.

        :param mean: the mean of the distribution
        :param std: the standard deviation
        :param lower: the lower limit of the distribution
        :param upper: the upper limit of the distribution
        :param precision: the precision of the returned number. if None then assumes int value else float
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
        precision = precision if isinstance(precision, int) else 3
        _seed = self._seed() if seed is None else seed
        rtn_list = stats.truncnorm((lower-mean)/std, (upper-mean)/std, loc=mean, scale=std).rvs(size).round(precision)
        return self._set_quantity(list(rtn_list), quantity=quantity, seed=_seed)

    def get_distribution(self, distribution: str, package: str=None, precision: int=None, size: int=None,
                         quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                         intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                         **kwargs) -> list:
        """returns a number based the distribution type.

        :param distribution: The string name of the distribution function from numpy random Generator class
        :param package: (optional) The name of the package to use, options are 'numpy' (default) and 'scipy'.
        :param precision: (optional) the precision of the returned number
        :param size: (optional) the size of the sample
        :param quantity: (optional) a number between 0 and 1 representing data that isn't null
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
        :param kwargs: the parameters of the method
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        precision = 3 if precision is None else precision
        if isinstance(package, str) and package == 'scipy':
            rtn_list = eval(f"stats.{distribution}.rvs(size=size, random_state=_seed, **kwargs)", globals(), locals())
        else:
            generator = np.random.default_rng(seed=_seed)
            rtn_list = eval(f"generator.{distribution}(size=size, **kwargs)", globals(), locals())
        rtn_list = list(rtn_list.round(precision))
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

        _seed = self._next_seed(_seed, seed)
        generator = np.random.default_rng(seed=_seed)
        rtn_list = []
        for c in list(pattern):
            if c in choices.keys():
                result = generator.choice(choices[c], size=size)
            elif not choice_only:
                result = [c]*size
            else:
                continue
            rtn_list = [i + j for i, j in zip(rtn_list, result)] if len(rtn_list) > 0 else result
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_selection(self, connector_name: str, column_header: str, relative_freq: list=None, sample_size: int=None,
                      selection_size: int=None, size: int=None, at_most: bool=None, shuffle: bool=None,
                      quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                      intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a random list of values where the selection of those values is taken from a connector source.

        :param connector_name: a connector_name for a connector to a data source
        :param column_header: the name of the column header to correlate
        :param relative_freq: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, normally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param at_most: (optional) the most times a selection should be chosen
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
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
        if column_header not in canonical.columns:
            raise ValueError(f"The column '{column_header}' not found in the data from connector '{connector_name}'")
        _values = canonical[column_header].iloc[:sample_size]
        if isinstance(selection_size, float) and shuffle:
            _values = _values.sample(frac=1, random_state=_seed).reset_index(drop=True)
        if isinstance(selection_size, int) and 0 < selection_size < _values.size:
            _values = _values.iloc[:selection_size]
        return self.get_category(selection=_values.to_list(), relative_freq=relative_freq, quantity=quantity,
                                 size=size, at_most=at_most, seed=_seed, save_intent=False)

    def get_sample(self, sample_name: str, sample_size: int=None, shuffle: bool=None, size: int=None,
                   quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                   intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ returns a sample set based on sector and name
        To see the sample sets available use the Sample class __dir__() method:

            > from ds_behavioral.sample.sample_data import Sample
            > Sample().__dir__()

        :param sample_name: The name of the Sample method to be used.
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
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
        :return: a sample list
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        size = 1 if size is None else size
        sample_size = sample_name if isinstance(sample_size, int) else size
        quantity = self._quantity(quantity)
        _seed = self._seed() if seed is None else seed
        shuffle = shuffle if isinstance(shuffle, bool) else True
        selection = eval(f"Sample.{sample_name}(size={size}, shuffle={shuffle}, seed={_seed})")
        return self._set_quantity(selection, quantity=quantity, seed=_seed)

    def get_uuid(self, version: int=None, as_hex: bool=None, size: int=None, quantity: float=None, seed: int=None,
                 save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                 replace_intent: bool=None, remove_duplicates: bool=None, **kwargs) -> list:
        """ returns a list of UUID's based on the version presented. By default the uuid version is 4. optional
        parameters for the version number UUID generator can be passed as kwargs.
        Version 1: Generate a UUID from a host ID, sequence number, and the current time. Note as uuid1 contains the
                   computers network address it may compromise privacy
                param node: (optional) used instead of getnode() which returns a hardware address
                param clock_seq: (optional) used as a sequence number alternative
        Version 3: Generate a UUID based on the MD5 hash of a namespace identifier and a name
                param namespace: an alternative namespace as a UUID e.g. uuid.NAMESPACE_DNS
                param name: a string name
        Version 4: Generate a random UUID
        Version 5: Generate a UUID based on the SHA-1 hash of a namespace identifier and name
                param namespace: an alternative namespace as a UUID e.g. uuid.NAMESPACE_DNS
                param name: a string name

        :param version: The version of the UUID to use. 1, 3, 4 or 5
        :param as_hex: if the return value is in hex format, else as a string
        :param size: the size of the sample. Must be smaller than the range
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
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        as_hex = as_hex if isinstance(as_hex, bool) else False
        version = version if isinstance(version, int) and version in [1, 3, 4, 5] else 4
        kwargs = kwargs if isinstance(kwargs, dict) else {}

        rtn_list = [eval(f'uuid{version}(**{kwargs})', globals(), locals()) for x in range(size)]
        rtn_list = [x.hex if as_hex else str(x) for x in rtn_list]
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_tagged_pattern(self, pattern: [str, list], tags: dict, relative_freq: list=None, size: int=None,
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
        :param relative_freq: a weighting pattern that does not have to add to 1
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
            choice = self.get_category(pattern, relative_freq=relative_freq, seed=_seed, size=1, save_intent=False)[0]
            for tag, action in tags.items():
                method = action.get('action')
                if method in class_methods:
                    kwargs = action.get('kwargs')
                    result = eval(f"self.{method}('save_intent=False, **{kwargs})")[0]
                else:
                    result = method
                choice = re.sub(tag, str(result), str(choice))
            rtn_list.append(choice)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def frame_selection(self, canonical: Any, selection: list=None, headers: [str, list]=None, drop: bool=None,
                        dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                        seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ Selects rows and/or columns changing the shape of the DatFrame. This is always run last in a pipeline
        Rows are filtered before the column filter so columns can be referenced even though they might not be included
        the final column list.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param seed: this is a place holder, here for compatibility across methods
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
        :return: pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        Selections are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'select2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [inst.select2dict(column='gender', condition="=='M'"),
                             inst.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed. It also
        helps with building the logic that is executed in order

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        if isinstance(selection, list):
            selection = deepcopy(selection)
            # run the select logic
            select_idx = self._selection_index(canonical=canonical, selection=selection)
            canonical = canonical.iloc[select_idx].reset_index(drop=True)
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        return SyntheticCommons.filter_columns(df=canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                               regex=regex, re_ignore_case=re_ignore_case)

    def model_iterator(self, canonical: Any, marker_col: str=None, starting_frame: str=None, selection: list=None,
                       default_action: dict=None, iteration_actions: dict=None, iter_start: int=None,
                       iter_stop: int=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ This method allows one to model repeating data subset that has some form of action applied per iteration.
        The optional marker column must be included in order to apply actions or apply an iteration marker
        An example of use might be a recommender generator where a cohort of unique users need to be selected, for
        different recommendation strategies but users can be repeated across recommendation strategy

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param marker_col: (optional) the marker column name for the action outcome. default is to not include
        :param starting_frame: (optional) a str referencing an existing connector contract name as the base DataFrame
        :param selection: (optional) a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param default_action: (optional) a default action to take on all iterations. defaults to iteration value
        :param iteration_actions: (optional) a dictionary of actions where the key is a specific iteration
        :param iter_start: (optional) the start value of the range iteration default is 0
        :param iter_stop: (optional) the stop value of the range iteration default is start iteration + 1
        :param seed: (optional) this is a place holder, here for compatibility across methods
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
        :return: pd.DataFrame


        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        Selections are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'select2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [inst.select2dict(column='gender', condition="=='M'"),
                             inst.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed. It also
        helps with building the logic that is executed in order

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection: ['M', 'F', 'U']}

        This same action using the helper method would look like:
                inst.action2dict(method='get_category', selection=['M', 'F', 'U'])

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        rtn_frame = self._get_canonical(starting_frame) if isinstance(starting_frame, str) else pd.DataFrame()
        _seed = self._seed() if seed is None else seed
        iter_start = iter_start if isinstance(iter_start, int) else 0
        iter_stop = iter_stop if isinstance(iter_stop, int) and iter_stop > iter_start else iter_start + 1
        default_action = default_action if isinstance(default_action, dict) else 0
        iteration_actions = iteration_actions if isinstance(iteration_actions, dict) else {}
        for counter in range(iter_start, iter_stop):
            df_count = canonical.copy()
            # selection
            df_count = self.frame_selection(df_count, selection=selection, seed=_seed, save_intent=False)
            # actions
            if isinstance(marker_col, str):
                if counter in iteration_actions.keys():
                    _action = iteration_actions.get(counter, None)
                    df_count[marker_col] = self._apply_action(df_count, action=_action, seed=_seed)
                else:
                    default_action = default_action if isinstance(default_action, dict) else counter
                    df_count[marker_col] = self._apply_action(df_count, action=default_action, seed=_seed)
            rtn_frame = pd.concat([rtn_frame, df_count], ignore_index=True)
        return rtn_frame

    def model_group(self, canonical: Any, headers: [str, list], group_by: [str, list], aggregator: str=None,
                    list_choice: int=None, list_max: int=None, drop_group_by: bool=False, seed: int=None,
                    include_weighting: bool=False, freq_precision: int=None, remove_weighting_zeros: bool=False,
                    remove_aggregated: bool=False, save_intent: bool=None, column_name: [int, str]=None,
                    intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source. in addition the the
        standard groupby aggregators there is also 'list' and 'set' that returns an aggregated list or set.
        These can be using in conjunction with 'list_choice' and 'list_size' allows control of the return values.
        if list_max is set to 1 then a single value is returned rather than a list of size 1.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param headers: the column headers to apply the aggregation too
        :param group_by: the column headers to group by
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby' or 'list' or 'set'
        :param list_choice: (optional) used in conjunction with list or set aggregator to return a random n choice
        :param list_max: (optional) used in conjunction with list or set aggregator restricts the list to a n size
        :param drop_group_by: (optional) drops the group by headers
        :param include_weighting: (optional) include a percentage weighting column for each
        :param freq_precision: (optional) a precision for the relative_freq values
        :param remove_aggregated: (optional) if used in conjunction with the weighting then drops the aggrigator column
        :param remove_weighting_zeros: (optional) removes zero values
        :param seed: (optional) this is a place holder, here for compatibility across methods
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
        :return: a pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        freq_precision = freq_precision if isinstance(freq_precision, int) else 3
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        headers = self._pm.list_formatter(headers)
        group_by = self._pm.list_formatter(group_by)
        df_sub = SyntheticCommons.filter_columns(canonical, headers=headers + group_by).dropna()
        if aggregator.startswith('set') or aggregator.startswith('list'):
            df_tmp = df_sub.groupby(group_by)[headers[0]].apply(eval(aggregator)).apply(lambda x: list(x))
            df_tmp = df_tmp.reset_index()
            for idx in range(1, len(headers)):
                result = df_sub.groupby(group_by)[headers[idx]].apply(eval(aggregator)).apply(lambda x: list(x))
                df_tmp = df_tmp.merge(result, how='left', left_on=group_by, right_index=True)
            for idx in range(len(headers)):
                header = headers[idx]
                if isinstance(list_choice, int):
                    df_tmp[header] = df_tmp[header].apply(lambda x: generator.choice(x, size=list_choice))
                if isinstance(list_max, int):
                    df_tmp[header] = df_tmp[header].apply(lambda x: x[0] if list_max == 1 else x[:list_max])

            df_sub = df_tmp
        else:
            df_sub = df_sub.groupby(group_by, as_index=False).agg(aggregator)
        if include_weighting:
            df_sub['sum'] = df_sub.sum(axis=1, numeric_only=True)
            total = df_sub['sum'].sum()
            df_sub['weighting'] = df_sub['sum'].\
                apply(lambda x: round((x / total), freq_precision) if isinstance(x, (int, float)) else 0)
            df_sub = df_sub.drop(columns='sum')
            if remove_weighting_zeros:
                df_sub = df_sub[df_sub['weighting'] > 0]
            df_sub = df_sub.sort_values(by='weighting', ascending=False)
        if remove_aggregated:
            df_sub = df_sub.drop(headers, axis=1)
        if drop_group_by:
            df_sub = df_sub.drop(columns=group_by, errors='ignore')
        return df_sub

    def model_merge(self, canonical: Any, other: [str, dict], left_on: str, right_on: str, how: str=None,
                    suffixes: tuple=None, indicator: bool=None, validate: str=None, seed: int=None,
                    save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                    replace_intent: bool=None,  remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param left_on: the canonical key column(s) to join on
        :param right_on: the merging dataset key column(s) to join on
        :param how: (optional) One of 'left', 'right', 'outer', 'inner'. Defaults to inner. See below for more detailed
                    description of each method.
        :param suffixes: (optional) A tuple of string suffixes to apply to overlapping columns. Defaults ('', '_dup').
        :param indicator: (optional) Add a column to the output DataFrame called _merge with information on the source
                    of each row. _merge is Categorical-type and takes on a value of left_only for observations whose
                    merge key only appears in 'left' DataFrame or Series, right_only for observations whose merge key
                    only appears in 'right' DataFrame or Series, and both if the observation’s merge key is found
                    in both.
        :param validate: (optional) validate : string, default None. If specified, checks if merge is of specified type.
                            “one_to_one” or “1:1”: checks if merge keys are unique in both left and right datasets.
                            “one_to_many” or “1:m”: checks if merge keys are unique in left dataset.
                            “many_to_one” or “m:1”: checks if merge keys are unique in right dataset.
                            “many_to_many” or “m:m”: allowed, but does not result in checks.
        :param seed: this is a place holder, here for compatibility across methods
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
        :return: a pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _seed = self._seed() if seed is None else seed
        how = how if isinstance(how, str) and how in ['left', 'right', 'outer', 'inner'] else 'inner'
        indicator = indicator if isinstance(indicator, bool) else False
        suffixes = suffixes if isinstance(suffixes, tuple) and len(suffixes) == 2 else ('', '_dup')
        # Filter on the columns
        df_rtn = pd.merge(left=canonical, right=other, how=how, left_on=left_on, right_on=right_on,
                          suffixes=suffixes, indicator=indicator, validate=validate)
        return df_rtn

    def model_concat(self, canonical: Any, other: Any, as_rows: bool=None, headers: [str, list]=None,
                     drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                     re_ignore_case: bool=None, shuffle: bool=None, seed: int=None, save_intent: bool=None,
                     column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param as_rows: (optional) how to concatenate, True adds the connector dataset as rows, False as columns
        :param headers: (optional) a filter of headers from the 'other' dataset
        :param drop: (optional) to drop or not drop the headers if specified
        :param dtype: (optional) a filter on data type for the 'other' dataset. int, float, bool, object
        :param exclude: (optional) to exclude or include the data types if specified
        :param regex: (optional) a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt'
        :param re_ignore_case: (optional) true if the regex should ignore case. Default is False
        :param shuffle: (optional) if the rows in the loaded canonical should be shuffled
        :param seed: this is a place holder, here for compatibility across methods
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
        :return: a pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _seed = self._seed() if seed is None else seed
        shuffle = shuffle if isinstance(shuffle, bool) else False
        as_rows = as_rows if isinstance(as_rows, bool) else False
        # Filter on the columns
        df_rtn = SyntheticCommons.filter_columns(df=other, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case, copy=False)
        if shuffle:
            df_rtn.sample(frac=1, random_state=_seed).reset_index(drop=True)
        if canonical.shape[0] <= df_rtn.shape[0]:
            df_rtn = df_rtn.iloc[:canonical.shape[0]]
        axis = 'index' if as_rows else 'columns'
        return pd.concat([canonical, df_rtn], axis=axis)

    def model_noise(self, canonical: Any, num_columns: int, inc_targets: bool=None, seed: int=None,
                    save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                    replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ Generates multiple columns of noise in your dataset

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param num_columns: the number of columns of noise
        :param inc_targets: (optional) if a predictor target should be included. default is false
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
        :return: a DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        _seed = self._seed() if seed is None else seed
        size = canonical.shape[0]
        num_columns = num_columns if isinstance(num_columns, int) else 1
        inc_targets = inc_targets if isinstance(inc_targets, int) else False
        gen = SyntheticCommons.label_gen()
        df_rtn = pd.DataFrame()
        generator = np.random.default_rng(seed=_seed)
        for _ in range(num_columns):
            _seed = self._next_seed(_seed, seed)
            a = generator.choice(range(1, 6))
            b = generator.choice(range(1, 6))
            df_rtn[next(gen)] = self.get_distribution(distribution='beta', a=a, b=b, precision=3, size=size, seed=_seed,
                                                      save_intent=False)
        if inc_targets:
            result = df_rtn.mean(axis=1)
            df_rtn['target1'] = result.apply(lambda x: 1 if x > 0.5 else 0)
            df_rtn['target2'] = df_rtn.iloc[:, :5].mean(axis=1).round(2)
        return pd.concat([canonical, df_rtn], axis=1)

    def model_sample_map(self, canonical: Any, sample_map: str, selection: list=None, headers: [str, list]=None,
                         shuffle: bool=None, rename_columns: dict=None, seed: int=None, save_intent: bool=None,
                         column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None, **kwargs) -> pd.DataFrame:
        """ builds a model of a Sample Mapped distribution.
        To see the sample maps available use the MappedSample class __dir__() method:

            > from ds_behavioral.sample.sample_data import MappedSample
            > MappedSample().__dir__()

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param sample_map: the sample map name. use MappedSample().__dir__() to get a list of available samples
        :param rename_columns: (optional) rename the columns 'City', 'Zipcode', 'State'
        :param selection: (optional) a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'state', 'condition': "isin(['NY', 'TX']"}]
        :param headers: a header or list of headers to filter on
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
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
        :param kwargs: any additional parameters to pass to the sample map
        :return: a DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        _seed = self._seed() if seed is None else seed
        shuffle = shuffle if isinstance(shuffle, bool) else True
        size = canonical.shape[0] if canonical.shape[0] > 1 else None
        df_rtn = eval(f"MappedSample.{sample_map}(size={size}, shuffle={shuffle}, seed={_seed}, **{kwargs})")
        if isinstance(headers, (list, str)):
            df_rtn = SyntheticCommons.filter_columns(df_rtn, headers=headers, copy=False)
        if isinstance(selection, list):
            selection = deepcopy(selection)
            # run the select logic
            select_idx = None
            select_idx = self._selection_index(canonical=df_rtn, selection=selection)
            df_rtn = df_rtn.iloc[select_idx]
        if isinstance(rename_columns, dict):
            df_rtn = df_rtn.rename(columns=rename_columns)
        return pd.concat([canonical, df_rtn], axis=1)

    def model_analysis(self, canonical: Any, analytics_model: dict, size: int=None, seed: int=None,
                       save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ builds a set of columns based on an analysis dictionary of weighting (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param analytics_model: the analytics model from discovery-transition-ds discovery model train
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
        :return: a DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)

        def get_level(analysis: dict, sample_size: int, _seed: int=None):
            _seed = self._next_seed(seed=_seed, default=seed)
            for name, values in analysis.items():
                if row_dict.get(name) is None:
                    row_dict[name] = list()
                _analysis = DataAnalytics(label=name, analysis=values.get('analysis', {}))
                if str(_analysis.intent.dtype).startswith('cat'):
                    row_dict[name] += self.get_category(selection=_analysis.intent.selection,
                                                        relative_freq=_analysis.patterns.relative_freq,
                                                        quantity=1-_analysis.stats.nulls_percent, seed=_seed,
                                                        size=sample_size, save_intent=False)
                if str(_analysis.intent.dtype).startswith('num'):
                    row_dict[name] += self.get_intervals(intervals=_analysis.intent.selection,
                                                         relative_freq=_analysis.patterns.relative_freq,
                                                         precision=_analysis.intent.precision,
                                                         quantity=1 - _analysis.stats.nulls_percent,
                                                         seed=_seed, size=sample_size, save_intent=False)
                if str(_analysis.intent.dtype).startswith('date'):
                    row_dict[name] += self.get_datetime(start=_analysis.intent.lower, until=_analysis.intent.upper,
                                                        relative_freq=_analysis.patterns.relative_freq,
                                                        date_format=_analysis.intent.data_format,
                                                        day_first=_analysis.intent.day_first,
                                                        year_first=_analysis.intent.year_first,
                                                        quantity=1 - _analysis.stats.nulls_percent,
                                                        seed=_seed, size=sample_size, save_intent=False)
                unit = sample_size / sum(_analysis.patterns.relative_freq)
                if values.get('sub_category'):
                    section_map = _analysis.relative_freq_map
                    for i in section_map.index:
                        section_size = int(round(_analysis.relative_freq_map.loc[i] * unit, 0)) + 1
                        next_item = values.get('sub_category').get(i)
                        get_level(next_item, section_size, _seed)
            return

        canonical = self._get_canonical(canonical)
        row_dict = dict()
        seed = self._seed() if seed is None else seed
        size = canonical.shape[0]
        get_level(analytics_model, sample_size=size, _seed=seed)
        for key in row_dict.keys():
            row_dict[key] = row_dict[key][:size]
        return pd.concat([canonical, pd.DataFrame.from_dict(data=row_dict)], axis=1)

    def correlate_selection(self, canonical: Any, selection: list, action: [str, int, float, dict],
                            default_action: [str, int, float, dict]=None, quantity: float=None, seed: int=None,
                            save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                            replace_intent: bool=None, remove_duplicates: bool=None):
        """ returns a value set based on the selection list and the action enacted on that selection. If
        the selection criteria is not fulfilled then the default_action is taken if specified, else null value.

        If a DataFrame is not passed, the values column is referenced by the header '_default'

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param action: a value or dict to act upon if the select is successful. see below for more examples
                An example of an action as a dict: (see 'action2dict(...)')
                {'method': 'get_category', 'selection': ['M', 'F', 'U']}
        :param default_action: (optional) a default action to take if the selection is not fulfilled
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
        :return: value set based on the selection list and the action

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        Selections are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'select2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [inst.select2dict(column='gender', condition="=='M'"),
                             inst.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed. It also
        helps with building the logic that is executed in order

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection: ['M', 'F', 'U']}

        This same action using the helper method would look like:
                inst.action2dict(method='get_category', selection=['M', 'F', 'U'])

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        if len(canonical) == 0:
            raise TypeError("The canonical given is empty")
        if not isinstance(selection, list):
            raise ValueError("The 'selection' parameter must be a 'list' of 'dict' types")
        if not isinstance(action, (str, int, float, dict)) or (isinstance(action, dict) and len(action) == 0):
            raise TypeError("The 'action' parameter is not of an acepted format or is empty")
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        # prep the values to be a DataFrame if it isn't already
        action = deepcopy(action)
        selection = deepcopy(selection)
        # run the logic
        select_idx = self._selection_index(canonical=canonical, selection=selection)
        if not isinstance(default_action, (str, int, float, dict)):
            default_action = None
        rtn_values = self._apply_action(canonical, action=default_action, seed=_seed)
        rtn_values.update(self._apply_action(canonical, action=action, select_idx=select_idx, seed=_seed))
        return self._set_quantity(rtn_values.to_list(), quantity=quantity, seed=_seed)

    def correlate_custom(self, canonical: Any, code_str: str, use_exec: bool=None, quantity: float=None, seed: int=None,
                         save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None, **kwargs):
        """ enacts an action on a dataFrame, returning the output of the action or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your action, it is internally referenced
        as it's parameter name 'canonical'.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
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
        :return: a list or pandas.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        use_exec = use_exec if isinstance(use_exec, bool) else False
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if result is None:
            return canonical
        return self._set_quantity(result, quantity=quantity, seed=_seed)

    def correlate_aggregate(self, canonical: Any, headers: list, agg: str, quantity: float=None, seed: int=None,
                            save_intent: bool=None, precision: int=None, column_name: [int, str]=None,
                            intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate two or more columns with each other through a finite set of aggregation functions. The
        aggregation function names are limited to 'sum', 'prod', 'count', 'min', 'max' and 'mean' for numeric columns
        and a special 'list' function name to combine the columns as a list

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param headers: a list of headers to correlate
        :param agg: the aggregation function name enact. The available functions are:
                        'sum', 'prod', 'count', 'min', 'max', 'mean' and 'list' which combines the columns as a list
        :param precision: the value precision of the return values
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
        :return: a list of equal length to the one passed

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # validation
        if not isinstance(headers, list) or len(headers) < 2:
            raise ValueError("The headers value must be a list of at least two header str")
        if agg not in ['sum', 'prod', 'count', 'min', 'max', 'mean', 'list']:
            raise ValueError("The only allowed func values are 'sum', 'prod', 'count', 'min', 'max', 'mean', 'list'")
        canonical = self._get_canonical(canonical)
        # Code block for intent
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        precision = precision if isinstance(precision, int) else 3
        if agg == 'list':
            result = canonical.loc[:, headers].values.tolist()
        else:
            result = eval(f"canonical.loc[:, headers].{agg}(axis=1)", globals(), locals()).round(precision)
        return self._set_quantity(result.to_list(), quantity=quantity, seed=_seed)

    def correlate_choice(self, canonical: Any, header: str, list_size: int=None, random_choice: bool=None,
                         replace: bool=None, shuffle: bool=None, convert_str: bool=None, quantity: float=None,
                         seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate a column where the elements of the columns contains a list, and a choice is taken from that list.
        if the list_size == 1 then a single value is correlated otherwise a list is correlated

        Null values are passed through but all other elements must be a list with at least 1 value in.

        if 'random' is true then all returned values will be a random selection from the list and of equal length.
        if 'random' is false then each list will not exceed the 'list_size'

        Also if 'random' is true and 'replace' is False then all lists must have more elements than the list_size.
        By default 'replace' is True and 'shuffle' is False.

        In addition 'convert_str' allows lists that have been formatted as a string can be converted from a string
        to a list using 'ast.literal_eval(x)'

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: The header containing a list to chose from.
        :param list_size: (optional) the number of elements to return, if more than 1 then list
        :param random_choice: (optional) if the choice should be a random choice.
        :param replace: (optional) if the choice selection should be replaced or selected only once
        :param shuffle: (optional) if the final list should be shuffled
        :param convert_str: if the header has the list as a string convert to list using ast.literal_eval()
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
        :return: a list of equal length to the one passed

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # validation
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        # Code block for intent
        list_size = list_size if isinstance(list_size, int) else 1
        random_choice = random_choice if isinstance(random_choice, bool) else False
        convert_str = convert_str if isinstance(convert_str, bool) else False
        replace = replace if isinstance(replace, bool) else True
        shuffle = shuffle if isinstance(shuffle, bool) else False
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        s_idx = s_values.where(~s_values.isna()).dropna().index
        if convert_str:
            s_values.iloc[s_idx] = [ast.literal_eval(x) if isinstance(x, str) else x for x in s_values.iloc[s_idx]]
        s_values.iloc[s_idx] = SyntheticCommons.list_formatter(s_values.iloc[s_idx])
        generator = np.random.default_rng(seed=_seed)
        if random_choice:
            try:
                s_values.iloc[s_idx] = [generator.choice(x, size=list_size, replace=replace, shuffle=shuffle)
                                        for x in s_values.iloc[s_idx]]
            except ValueError:
                raise ValueError(f"Unable to make a choice. Ensure {header} has all appropriate values for the method")
            s_values.iloc[s_idx] = [x[0] if list_size == 1 else list(x) for x in s_values.iloc[s_idx]]
        else:
            s_values.iloc[s_idx] = [x[:list_size] if list_size > 1 else x[0] for x in s_values.iloc[s_idx]]
        return self._set_quantity(s_values.to_list(), quantity=quantity, seed=_seed)

    def correlate_join(self, canonical: Any, header: str, action: [str, dict], sep: str=None, quantity: float=None,
                       seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate a column and join it with the result of the action

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: an ordered list of columns to join
        :param action: (optional) a string or a single action whose outcome will be joined to the header value
        :param sep: (optional) a separator between the values
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
        :return: a list of equal length to the one passed

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection=['M', 'F', 'U']

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # validation
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(action, (dict, str)):
            raise ValueError(f"The action must be a dictionary of a single action or a string value")
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        # Code block for intent
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        sep = sep if isinstance(sep, str) else ''
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        action = deepcopy(action)
        null_idx = s_values[s_values.isna()].index
        s_values.to_string()
        result = self._apply_action(canonical, action=action, seed=_seed)
        s_values = pd.Series([f"{a}{sep}{b}" for (a, b) in zip(s_values, result)])
        if null_idx.size > 0:
            s_values.iloc[null_idx] = np.nan
        return self._set_quantity(s_values.to_list(), quantity=quantity, seed=_seed)

    def correlate_sigmoid(self, canonical: Any, header: str, precision: int=None, quantity: float=None, seed: int=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None):
        """ logistic sigmoid a.k.a logit, takes an array of real numbers and transforms them to a value
        between (0,1) and is defined as
                                        f(x) = 1/(1+exp(-x)

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param precision: (optional) how many decimal places. default to 3
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) the random seed used with quantity. defaults to current datetime
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        precision = precision if isinstance(precision, int) else 3
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        result = np.round(1 / (1 + np.exp(-s_values)), precision)
        return self._set_quantity(list(result), quantity=quantity, seed=_seed)

    def correlate_polynomial(self, canonical: Any, header: str, coefficient: list, quantity: float=None,
                             seed: int=None, keep_zero: bool=None, save_intent: bool=None, column_name: [int, str]=None,
                             intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ creates a polynomial using the reference header values and apply the coefficients where the
        index of the list represents the degree of the term in reverse order.

                  e.g  [6, -2, 0, 4] => f(x) = 4x**3 - 2x + 6

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param coefficient: the reverse list of term coefficients
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) the random seed used with quantity. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        keep_zero = keep_zero if isinstance(keep_zero, bool) else False
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()

        def _calc_polynomial(x, _coefficient):
            if keep_zero and x == 0:
                return 0
            res = 0
            for index, coeff in enumerate(_coefficient):
                res += coeff * x ** index
            return res

        result = s_values.apply(lambda x: _calc_polynomial(x, coefficient))
        return self._set_quantity(result.to_list(), quantity=quantity, seed=_seed)

    def correlate_numbers(self, canonical: Any, header: str, offset: float=None, jitter: float=None,
                          jitter_freq: list=None, multiply_offset: bool=None, precision: int=None,
                          fill_nulls: bool=None, quantity: float=None, seed: int=None, keep_zero: bool=None,
                          min_value: [int, float]=None, max_value: [int, float]=None, save_intent: bool=None,
                          column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ returns a number that correlates to the value given. The jitter is based on a normal distribution
        with the correlated value being the mean and the jitter its standard deviation from that mean

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param offset: (optional) how far from the value to offset. defaults to zero
        :param jitter: (optional) a perturbation of the value where the jitter is a std. defaults to 0
        :param jitter_freq: (optional)  a relative freq with the pattern mid point the mid point of the jitter
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical, header=header)
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
            generator = np.random.default_rng(seed=_seed)
            s_values = s_values.fillna(generator.choice(s_values.mode(dropna=True)))
        null_idx = s_values[s_values.isna()].index
        zero_idx = s_values.where(s_values == 0).dropna().index if keep_zero else []
        if isinstance(offset, (int, float)) and offset != 0:
            s_values = s_values.mul(offset) if action == 'multiply' else s_values.add(offset)
        if isinstance(jitter, (int, float)) and jitter != 0:
            sample = self.get_number(-abs(jitter) / 2, abs(jitter) / 2, relative_freq=jitter_freq,
                                     size=s_values.size, save_intent=False, seed=_seed)
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
        return self._set_quantity(s_values.to_list(), quantity=quantity, seed=_seed)

    def correlate_categories(self, canonical: Any, header: str, correlations: list, actions: dict,
                             default_action: [str, int, float, dict]=None, quantity: float=None,
                             seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                             intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
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

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param correlations: a list of categories (can also contain lists for multiple correlations.
        :param actions: the correlated set of categories that should map to the index
        :param default_action: (optional) a default action to take if the selection is not fulfilled
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


        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection=['M', 'F', 'U']

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        quantity = self._quantity(quantity)
        _seed = seed if isinstance(seed, int) else self._seed()
        actions = deepcopy(actions)
        correlations = deepcopy(correlations)
        corr_list = []
        for corr in correlations:
            corr_list.append(self._pm.list_formatter(corr))
        if not isinstance(default_action, (str, int, float, dict)):
            default_action = None
        rtn_values = self._apply_action(canonical, action=default_action, seed=_seed)
        s_values = canonical[header].copy().astype(str)
        for i in range(len(corr_list)):
            action = actions.get(i, actions.get(str(i), -1))
            if action == -1:
                continue
            corr_idx = s_values[s_values.isin(map(str, corr_list[i]))].index
            rtn_values.update(self._apply_action(canonical, action=action, select_idx=corr_idx, seed=_seed))
        return self._set_quantity(rtn_values.to_list(), quantity=quantity, seed=_seed)

    def correlate_dates(self, canonical: Any, header: str, offset: [int, dict]=None, jitter: int=None,
                        jitter_units: str=None, jitter_freq: list=None, now_delta: str=None, date_format: str=None,
                        min_date: str=None, max_date: str=None, fill_nulls: bool=None, day_first: bool=None,
                        year_first: bool=None, quantity: float=None, seed: int=None, save_intent: bool=None,
                        column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None):
        """ correlates dates to an existing date or list of dates. The return is a list of pd

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param offset: (optional) and offset to the date. if int then assumed a 'days' offset
                int or dictionary associated with pd. eg {'days': 1}
        :param jitter: (optional) the random jitter or deviation in days
        :param jitter_units: (optional) the units of the jitter, Options: 'W', 'D', 'h', 'm', 's'. default 'D'
        :param jitter_freq: (optional) a relative freq with the pattern mid point the mid point of the jitter
        :param now_delta: (optional) returns a delta from now as an int list, Options: 'Y', 'M', 'W', 'D', 'h', 'm', 's'
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical, header=header)
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
        if isinstance(now_delta, str) and now_delta not in ['Y', 'M', 'W', 'D', 'h', 'm', 's']:
            raise ValueError(f"the now_delta offset unit '{now_delta}' is not recognised "
                             f"use of of ['Y', 'M', 'W', 'D', 'h', 'm', 's']")
        units_allowed = ['W', 'D', 'h', 'm', 's']
        jitter_units = jitter_units if isinstance(jitter_units, str) and jitter_units in units_allowed else 'D'
        jitter = pd.Timedelta(value=jitter, unit=jitter_units) if isinstance(jitter, int) else None
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
        if jitter is not None:
            if jitter_units in ['W', 'D']:
                value = jitter.days
                zip_units = 'D'
            else:
                value = int(jitter.to_timedelta64().astype(int) / 1000000000)
                zip_units = 's'
            zip_spread = self.get_number(-abs(value) / 2, (abs(value+1) / 2), relative_freq=jitter_freq,
                                         precision=0, size=s_values.size, save_intent=False, seed=_seed)
            zipped_dt = list(zip(zip_spread, [zip_units]*s_values.size))
            s_values = s_values + np.array([pd.Timedelta(x, y).to_timedelta64() for x, y in zipped_dt])
        if fill_nulls:
            generator = np.random.default_rng(seed=_seed)
            s_values = s_values.fillna(generator.choice(s_values.mode(dropna=True)))
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
        if now_delta:
            s_values = (s_values.dt.tz_convert(None) - pd.Timestamp('now')).abs()
            s_values = (s_values / np.timedelta64(1, now_delta))
            s_values = s_values.round(0) if null_idx.size > 0 else s_values.astype(int)
        else:
            if isinstance(date_format, str):
                s_values = s_values.dt.strftime(date_format)
            else:
                s_values = s_values.dt.tz_convert(None)
            if null_idx.size > 0:
                s_values.iloc[null_idx].apply(lambda x: np.nan)
        return self._set_quantity(s_values.to_list(), quantity=quantity, seed=_seed)

    """
        UTILITY METHODS SECTION
    """

    @property
    def sample_lists(self) -> list:
        """A list of sample options"""
        return Sample().__dir__()

    @property
    def sample_maps(self) -> list:
        """A list of sample options"""
        return MappedSample().__dir__()

    @staticmethod
    def select2dict(column: str, condition: str, expect: str=None, logic: str=None, date_format: str=None,
                    offset: int=None) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.

        :param column: the column name to apply the condition to
        :param condition: the condition string (special conditions are 'date.now' for current date
        :param expect: (optional) the data type to expect. If None then the data type is assumed from the dtype
        :param logic: (optional) the logic to provide, see below for options
        :param date_format: (optional) a format of the date if only a specific part of the date and time is required
        :param offset: (optional) a time delta in days (+/-) from the current date and time (minutes not supported)
        :return: dictionary of the parameters

        logic:
            AND: the intersect of the current state with the condition result (common to both)
            NAND: outside the intersect of the current state with the condition result (not common to both)
            OR: the union of the current state with the condition result (everything in both)
            NOR: outside the union of the current state with the condition result (everything not in both)
            NOT: the difference between the current state and the condition result
            XOR: the difference between the union and the intersect current state with the condition result
        extra logic:
            ALL: the intersect of the whole index with the condition result irrelevant of level or current state index
            ANY: the intersect of the level index with the condition result irrelevant of current state index
        """
        return SyntheticCommons.param2dict(**locals())

    @staticmethod
    def action2dict(method: Any, **kwargs) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.

        :param method: the method to execute
        :param kwargs: name value pairs associated with the method
        :return: dictionary of the parameters

        Special method values
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required


        """
        return SyntheticCommons.param2dict(method=method, **kwargs)

    @staticmethod
    def canonical2dict(method: Any, **kwargs) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.
        The method parameter can be wither a 'model_*' or 'frame_*' method with two special reserved options

        Special reserved method values
            @empty: returns an empty dataframe, optionally the key values size: int and headers: list
            @generate: generates a dataframe either from_env(task_name) o from a remote repo uri. params are
                task_name: the task name of the generator
                repo_uri: (optional) a remote repo to retrieve the the domain contract
                size: (optional) the generated sample size
                seed: (optional) if seeding should be applied the seed value
                run_book: (optional) a domain contract runbook to execute as part of the pipeline

        :param method: the method to execute
        :param kwargs: name value pairs associated with the method
        :return: dictionary of the parameters
        """
        return SyntheticCommons.param2dict(method=method, **kwargs)

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

    def _apply_action(self, canonical: pd.DataFrame, action: Any, select_idx: pd.Int64Index=None, seed: int=None):
        """ applies an action returning an indexed Series
        Special method values
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        :param canonical: a reference canonical
        :param action: the action dictionary
        :param select_idx: (optional) the index selection of the return Series. if None then canonical index taken
        :param seed: a seed to apply to any action
        :return: pandas Series with passed index
        """
        if not isinstance(select_idx, pd.Int64Index):
            select_idx = canonical.index
        if isinstance(action, dict):
            method = action.pop('method', None)
            if method is None:
                raise ValueError(f"The action dictionary has no 'method' key.")
            if method in self.__dir__():
                if isinstance(seed, int) and 'seed' not in action.keys():
                    action.update({'seed': seed})
                if str(method).startswith('get_'):
                    action.update({'size': select_idx.size, 'save_intent': False})
                    result = eval(f"self.{method}(**action)", globals(), locals())
                elif str(method).startswith('correlate_'):
                    action.update({'canonical': canonical.iloc[select_idx], 'save_intent': False})
                    result = eval(f"self.{method}(**action)", globals(), locals())
                else:
                    raise NotImplementedError(f"The method {method} is not implemented as part of the actions")
                dtype = 'object'
                if any(isinstance(x, int) for x in result):
                    dtype = 'int'
                elif any(isinstance(x, float) for x in result):
                    dtype = 'float'
                return pd.Series(data=result, index=select_idx, dtype=dtype)
            elif str(method).startswith('@header'):
                header = action.pop('header', None)
                if header is None:
                    raise ValueError(f"The action '@header' requires a 'header' key.")
                if header not in canonical.columns:
                    raise ValueError(f"When executing the action '@header', the header {header} was not found")
                return canonical[header].iloc[select_idx]
            elif str(method).startswith('@choice'):
                header = action.pop('header', None)
                size = action.pop('size', 1)
                size = action.pop('seed', 1)
                if header is None:
                    raise ValueError(f"The action '@choice' requires a 'header' key")
                if header not in canonical.columns:
                    raise ValueError(f"When executing the action '@choice', the header {header} was not found")
                generator = np.random.default_rng(seed=seed)
                return pd.Series([np.random.choice(x) for x in canonical[header]]).iloc[select_idx]
            elif str(method).startswith('@eval'):
                code_str = action.pop('code_str', None)
                if code_str is None:
                    raise ValueError(f"The action '@eval' requires a 'code_str' key.")
                e_value = eval(code_str, globals(), action)
                return pd.Series(data=([e_value] * select_idx.size), index=select_idx)
            elif str(method).startswith('@constant'):
                constant = action.pop('value', None)
                if constant is None:
                    raise ValueError(f"The action '@constant' requires a 'value' key.")
                return pd.Series(data=([constant] * select_idx.size), index=select_idx)
            elif str(method).startswith('@sample'):
                name = action.pop('name', None)
                shuffle = action.pop('shuffle', True)
                if name is None:
                    raise ValueError(f"The action '@sample' requires the 'name' key.")
                sample = eval(f'Sample.{name}(size={select_idx.size}, shuffle={shuffle}, seed={seed})')
                return pd.Series(data=sample, index=select_idx)
            else:
                raise ValueError(f"The 'method' key {method} is not a recognised intent method")
        return pd.Series(data=([action] * select_idx.size), index=select_idx)

    def _selection_index(self, canonical: pd.DataFrame, selection: list, select_idx: pd.Index=None):
        """ private method to iterate a list of selections and return the resulting index

        :param canonical: a pandas DataFrame to select from
        :param selection: the selection list of dictionaries
        :param select_idx: a starting index, if None then canonical index used
        :return:
        """
        select_idx = select_idx if isinstance(select_idx, pd.Index) else canonical.index
        sub_select_idx = select_idx
        state_idx = None
        for condition in selection:
            # print("\nCONDITION")
            # print(f"  {condition}")
            # print(f"  select_idx: {select_idx}")
            # print(f"  sub_select_idx: {sub_select_idx}")
            # print(f"  state_idx: {state_idx}")
            if isinstance(condition, str):
                condition = {'logic': condition}
            if isinstance(condition, dict):
                if not isinstance(state_idx, pd.Index):
                    state_idx = sub_select_idx
                if len(condition) == 1 and 'logic' in condition.keys():
                    if condition.get('logic') == 'ALL':
                        condition_idx = canonical.index
                    elif condition.get('logic') == 'ANY':
                        condition_idx = sub_select_idx
                    elif condition.get('logic') == 'NOT':
                        condition_idx = state_idx
                        state_idx = sub_select_idx
                    else:
                        condition_idx = state_idx
                else:
                    condition_idx = self._condition_index(canonical=canonical.iloc[sub_select_idx], condition=condition,
                                                          select_idx=sub_select_idx)
                logic = condition.get('logic', 'AND')
                state_idx = self._condition_logic(base_idx=canonical.index, sub_select_idx=sub_select_idx,
                                                  state_idx=state_idx, condition_idx=condition_idx, logic=logic)
                # print(f"  result: {state_idx}")
            elif isinstance(condition, list):
                if not isinstance(state_idx, pd.Index) or len(state_idx) == 0:
                    state_idx = sub_select_idx
                state_idx = self._selection_index(canonical=canonical, selection=condition, select_idx=state_idx)
            else:
                raise ValueError(f"The subsection of the selection list {condition} is neither a dict or a list")
        return state_idx

    @staticmethod
    def _condition_logic(base_idx: pd.Index, sub_select_idx: pd.Index, state_idx: pd.Index, condition_idx: pd.Index,
                         logic: str) -> pd.Index:
        # print("\n  LOGIC")
        # print(f"    sub_select_idx: {sub_select_idx}")
        # print(f"    state_idx: {state_idx}")
        # print(f"    condition_idx: {condition_idx}")
        if str(logic).upper() == 'ALL':
            return base_idx.intersection(condition_idx).sort_values()
        elif str(logic).upper() == 'ANY':
            return sub_select_idx.intersection(condition_idx).sort_values()
        elif str(logic).upper() == 'AND':
            return state_idx.intersection(condition_idx).sort_values()
        elif str(logic).upper() == 'NAND':
            return sub_select_idx.drop(state_idx.intersection(condition_idx)).sort_values()
        elif str(logic).upper() == 'OR':
            return state_idx.append(state_idx.union(condition_idx)).drop_duplicates(keep='first').sort_values()
        elif str(logic).upper() == 'NOR':
            result = state_idx.append(state_idx.union(condition_idx)).drop_duplicates(keep='first').sort_values()
            return sub_select_idx.drop(result)
        elif str(logic).upper() == 'NOT':
            return state_idx.difference(condition_idx)
        elif str(logic).upper() == 'XOR':
            return state_idx.union(condition_idx).difference(state_idx.intersection(condition_idx))
        raise ValueError(f"The logic '{logic}' must be AND, NAND, OR, NOR, NOT, XOR ANY or ALL")

    @staticmethod
    def _condition_index(canonical: pd.DataFrame, condition: dict, select_idx: pd.Index) -> pd.Index:
        _column = condition.get('column')
        _condition = condition.get('condition')
        if _column == '@':
            _condition = str(_condition).replace("@", "canonical.iloc[select_idx]")
        elif _column not in canonical.columns:
            raise ValueError(f"The column name '{_column}' can not be found in the canonical headers.")
        else:
            _condition = str(_condition).replace("@", f"canonical['{_column}']")
        # find the selection index
        return eval(f"canonical[{_condition}].index", globals(), locals())

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

    @staticmethod
    def _freq_dist_size(relative_freq: list, size: int, seed: int=None):
        """ utility method taking a list of relative frequencies and based on size returns the size distribution
        of element based on the frequency. The distribution is based upon binomial distributions

        :param relative_freq: a list of int or float values representing a relative distribution frequency
        :param size: the size to be distributed
        :param seed: (optional) a seed value for the random function: default to None
        :return: an integer list of the distrubution that sum to the size
        """
        if not isinstance(relative_freq, list) or not all(isinstance(x, (int, float)) for x in relative_freq):
            raise ValueError("The weighted pattern must be an list of numbers")
        seed = seed if isinstance(seed, int) else int(time.time() * np.random.random())
        if sum(relative_freq) != 1:
            relative_freq = np.round(relative_freq / np.sum(relative_freq), 5)
        generator = np.random.default_rng(seed=seed)
        result = list(generator.binomial(n=size, p=relative_freq, size=len(relative_freq)))
        diff = size - sum(result)
        adjust = [0] * len(relative_freq)
        if diff != 0:
            unit = diff / sum(relative_freq)
            for idx in range(len(relative_freq)):
                adjust[idx] = int(round(relative_freq[idx] * unit, 0))
        result = [a + b for (a, b) in zip(result, adjust)]
        # There is a possibility the required size is not fulfilled, therefore add or remove elements based on freq

        def _freq_choice(p: list):
            """returns a single index of the choice of the relative frequency"""
            rnd = generator.random() * sum(p)
            for i, w in enumerate(p):
                rnd -= w
                if rnd < 0:
                    return i

        while sum(result) != size:
            if sum(result) < size:
                result[_freq_choice(relative_freq)] += 1
            else:
                weight_idx = _freq_choice(relative_freq)
                if result[weight_idx] > 0:
                    result[weight_idx] -= 1
        return result

    def _set_quantity(self, selection, quantity, seed=None):
        """Returns the quantity percent of good values in selection with the rest fill"""
        if quantity == 1:
            return selection
        seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=seed)

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
                if generator.random() > quantity:
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
            seed = default if isinstance(default, int) else self._seed()
        np.random.seed(seed)
        return seed

    def _get_canonical(self, data: [pd.DataFrame, pd.Series, list, str, dict], header: str=None, size: int=None,
                       deep_copy: bool=None) -> pd.DataFrame:
        """ Used to return or generate a pandas Dataframe from a number of different methods.
        The following can be passed and their returns
        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use the canonical2dict(...) method to construct a dict with a method and related parameters
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and paramters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        :param data: a dataframe or action event to generate a dataframe
        :param header: (optional) header for pd.Series or list
        :param size: (optional) a size parameter for @empty of @generate
        :param header: (optional) used in conjunction with lists or pd.Series to give a header reference
        :return: a pd.Dataframe
        """
        deep_copy = deep_copy if isinstance(deep_copy, bool) else True
        if isinstance(data, pd.DataFrame):
            if deep_copy:
                return deepcopy(data)
            return data
        if isinstance(data, dict):
            method = data.pop('method', None)
            if method is None:
                raise ValueError(f"The data dictionary has no 'method' key.")
            if method in self.__dir__():
                if str(method).startswith('model_') or str(method).startswith('frame_'):
                    data.update({'save_intent': False})
                    return eval(f"self.{method}(**data)", globals(), locals())
            elif str(method).startswith('@generate'):
                task_name = data.pop('task_name', None)
                if task_name is None:
                    raise ValueError(f"The data method '@generate' requires a 'task_name' key.")
                uri_pm_repo = data.pop('uri_pm_repo', None)
                module = HandlerFactory.get_module(module_name='ds_behavioral')
                inst = module.SyntheticBuilder.from_env(task_name=task_name, uri_pm_repo=uri_pm_repo,
                                                        default_save=False)
                size = size if isinstance(size, int) and 'size' not in data.keys() else data.pop('size', None)
                seed = data.get('seed', None)
                run_book = data.pop('run_book', None)
                result = inst.tools.run_intent_pipeline(size=size, columns=run_book, seed=seed)
                return inst.tools.frame_selection(canonical=result, save_intent=False, **data)
            elif str(method).startswith('@empty'):
                size = size if isinstance(size, int) and 'size' not in data.keys() else data.pop('size', None)
                headers = data.pop('headers', None)
                size = range(size) if size else None
                return pd.DataFrame(index=size, columns=headers)
            else:
                raise ValueError(f"The data 'method' key {method} is not a recognised intent method")
        elif isinstance(data, (list, pd.Series)):
            header = header if isinstance(header, str) else 'default'
            if deep_copy:
                data = deepcopy(data)
            return pd.DataFrame(data=data, columns=[header])
        elif isinstance(data, str):
            if data == '@empty':
                return pd.DataFrame()
            if not self._pm.has_connector(connector_name=data):
                raise ValueError(f"The data connector name '{data}' is not in the connectors catalog")
            handler = self._pm.get_connector_handler(data)
            canonical = handler.load_canonical()
            if isinstance(canonical, dict):
                canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
            return canonical
        raise ValueError(f"The canonical format is not recognised, pd.DataFrame, pd.Series"
                         f"str, list or dict expected, {type(data)} passed")
