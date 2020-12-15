import re
import threading
from copy import deepcopy
import pandas as pd
import numpy as np
from aistac.components.aistac_commons import AistacCommons, AnalyticsCommons

__author__ = 'Darryl Oatridge'


class SyntheticCommons(AistacCommons):

    @staticmethod
    def exponential_tail(interval: float=None, precision: int=None, reverse: bool=False):
        """ returns an exponential tail with values decenting for tailing off weighting patterns.
        The tail values are generally between 0.9~ and 0.01~ with variance in length dependant on interval

        :param interval: the interval value for the exponential values determing the length of the tail. default 0.3
        :param precision: the precision of the values. default 3
        :param reverse: if the exponential should be reversed
        :return:
        """
        interval = interval if isinstance(interval, float) and interval < 1 else 0.3
        precision = precision if isinstance(precision, int) else 3
        result = np.exp(np.arange(-5, 0.0, interval)).round(precision)
        if isinstance(reverse, bool) and reverse:
            return list(result)
        return list(np.flip(result))

    @staticmethod
    def report(canonical: pd.DataFrame, index_header: [str, list], bold: [str, list]=None, large_font: [str, list]=None):
        """ generates a stylised report

        :param canonical
        :param index_header:
        :param bold:
        :param large_font
        :return: stylised report DataFrame
        """
        bold = SyntheticCommons.list_formatter(bold).append(index_header)
        large_font = SyntheticCommons.list_formatter(large_font).append(index_header)
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        index = canonical[canonical[index_header].duplicated()].index.to_list()
        canonical.loc[index, index_header] = ''
        canonical = canonical.reset_index(drop=True)
        df_style = canonical.style.set_table_styles(style)
        _ = df_style.set_properties(**{'text-align': 'left'})
        _ = df_style.set_properties(subset=bold, **{'font-weight': 'bold'})
        _ = df_style.set_properties(subset=large_font, **{'font-size': "120%"})
        return df_style

    @staticmethod
    def fillna(df: pd.DataFrame):
        """replaces NaN values - 0 for int, float, datetime, <NA> for category, False for bool, '' for objects """
        for col in df.columns:
            if df[col].dtype.name.lower().startswith('int') or df[col].dtype.name.startswith('float'):
                df[col].fillna(0, inplace=True)
            elif df[col].dtype.name.lower().startswith('date') or df[col].dtype.name.lower().startswith('time'):
                df[col].fillna(0, inplace=True)
            elif df[col].dtype.name.lower().startswith('bool'):
                df[col].fillna(False, inplace=True)
            elif df[col].dtype.name.lower().startswith('category'):
                df[col] = df[col].cat.add_categories("<NA>").fillna('<NA>')
            else:
                df[col].fillna('', inplace=True)
        return df

    @staticmethod
    def list_formatter(value) -> list:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series keys() etc into a list"""
        if isinstance(value, pd.Timestamp):
            return [value]
        if isinstance(value, pd.Series):
            return value.tolist()
        return AistacCommons.list_formatter(value=value)

    @staticmethod
    def filter_headers(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                       exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None) -> list:
        """ returns a list of headers based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None.
                    example: int, float, bool, 'category', 'object', 'number'. 'datetime', 'datetimetz', 'timedelta'
        :param exclude: to exclude or include the dtypes. Default is False
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :return: a filtered list of headers

        :raise: TypeError if any of the types are not as expected
        """
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The first function attribute must be a pandas 'DataFrame'")
        _headers = SyntheticCommons.list_formatter(headers)
        dtype = SyntheticCommons.list_formatter(dtype)
        regex = SyntheticCommons.list_formatter(regex)
        _obj_cols = df.columns
        _rtn_cols = set()
        unmodified = True

        if _headers:
            _rtn_cols = set(_obj_cols).difference(_headers) if drop else set(_obj_cols).intersection(_headers)
            unmodified = False

        if regex and regex:
            re_ignore_case = re.I if re_ignore_case else 0
            _regex_cols = list()
            for exp in regex:
                _regex_cols += [s for s in _obj_cols if re.search(exp, s, re_ignore_case)]
            _rtn_cols = _rtn_cols.union(set(_regex_cols))
            unmodified = False

        if unmodified:
            _rtn_cols = set(_obj_cols)

        if dtype and len(dtype) > 0:
            _df_selected = df.loc[:, _rtn_cols]
            _rtn_cols = (_df_selected.select_dtypes(exclude=dtype) if exclude
                         else _df_selected.select_dtypes(include=dtype)).columns

        return [c for c in _rtn_cols]

    @staticmethod
    def filter_columns(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                       exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                       copy: bool=None) -> pd.DataFrame:
        """ Returns a subset of columns based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param copy: if the passed pandas.DataFrame should be used or a deep copy. Default is True
        :return:
        """
        copy = copy if isinstance(copy, bool) else True
        if copy:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = SyntheticCommons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                   regex=regex, re_ignore_case=re_ignore_case)
        return df.loc[:, obj_cols]

    @staticmethod
    def col_width(df, column, as_value=False) -> tuple:
        """ gets the max and min length or values of a column as a (max, min) tuple

        :param df: the pandas.DataFrame
        :param column: the column to find the max and min for
        :param as_value: if the return should be a length or the values. Default is False
        :return: returns a tuple with the (max, min) length or values
        """
        if as_value:
            field_length = df[column].apply(str).str.len()
            return df.loc[field_length.argmax(), column], df.loc[field_length.argmin(), column]
        return df[column].apply(str).str.len().max(), df[column].apply(str).str.len().min()

    @staticmethod
    def data_schema(report: dict, stylise: bool=True):
        """ returns the schema dictionary as the canonical with optional style"""
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                     {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
            pd.set_option('max_colwidth', 200)
            df_style = df.style.set_table_styles(style)
            _ = df_style.applymap(SyntheticCommons._highlight_null_dom, subset=['%_Null', '%_Dom'])
            _ = df_style.applymap(lambda x: 'color: white' if x > 0.98 else 'color: black', subset=['%_Null', '%_Dom'])
            if '%_Nxt' in df.columns:
                _ = df_style.applymap(SyntheticCommons._highlight_next, subset=['%_Nxt'])
                _ = df_style.applymap(lambda x: 'color: white' if x < 0.02 else 'color: black', subset=['%_Nxt'])
            _ = df_style.applymap(SyntheticCommons._dtype_color, subset=['dType'])
            _ = df_style.applymap(SyntheticCommons._color_unique, subset=['Unique'])
            _ = df_style.applymap(lambda x: 'color: white' if x < 2 else 'color: black', subset=['Unique'])
            _ = df_style.format({'%_Null': "{:.1%}", '%_Dom': '{:.1%}', '%_Nxt': '{:.1%}'})
            _ = df_style.set_caption('%_Dom: The % most dominant element - %_Nxt: The % next most dominant element')
            _ = df_style.set_properties(subset=[next(k for k in report.keys() if 'Attributes' in k)],
                                        **{'font-weight': 'bold', 'font-size': "120%"})
            return df_style
        return df

    @staticmethod
    def data_dictionary(df, stylise: bool=False, inc_next_dom: bool=False, report_header: str=None,
                        condition: str=None):
        """ returns a DataFrame of a data dictionary showing 'Attribute', 'Type', '% Nulls', 'Count',
        'Unique', 'Observations' where attribute is the column names in the df
        Note that the subject_matter, if used, should be in the form:
            { subject_ref, { column_name : text_str}}
        the subject reference will be the header of the column and the text_str put in next to each attribute row

        :param df: (optional) the pandas.DataFrame to get the dictionary from
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param inc_next_dom: (optional) if to include the next dominate element column
        :param report_header: (optional) filter on a header where the condition is true. Condition must exist
        :param condition: (optional) the condition to apply to the header. Header must exist. examples:
                ' > 0.95', ".str.contains('shed')"
        :return: a pandas.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        inc_next_dom = False if not isinstance(inc_next_dom, bool) else inc_next_dom
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        pd.set_option('max_colwidth', 200)
        df_len = len(df)
        file = []
        labels = [f'Attributes ({len(df.columns)})', 'dType', '%_Null', '%_Dom', '%_Nxt', 'Count', 'Unique',
                  'Observations']
        for c in df.columns.sort_values().values:
            line = [c,
                    str(df[c].dtype),
                    round(df[c].replace('', np.nan).isnull().sum() / df_len, 3)]
            # Predominant Difference
            col = deepcopy(df[c])
            if len(col.dropna()) > 0:
                result = (col.apply(str).value_counts() /
                          np.float(len(col.apply(str).dropna()))).sort_values(ascending=False).values
                line.append(round(result[0], 3))
                if len(result) > 1:
                    line.append(round(result[1], 3))
                else:
                    line.append(0)
            else:
                line.append(0)
                line.append(0)
            # value count
            line.append(col.apply(str).notnull().sum())
            # unique
            line.append(col.apply(str).nunique())
            # Observations
            if col.dtype.name == 'category' or col.dtype.name == 'object' or col.dtype.name == 'string':
                value_set = list(col.dropna().apply(str).value_counts().index)
                if len(value_set) > 0:
                    sample_num = 5 if len(value_set) >= 5 else len(value_set)
                    sample = str(' | '.join(value_set[:sample_num]))
                else:
                    sample = 'Null Values'
                line_str = 'Sample: {}'.format(sample)
                line.append('{}...'.format(line_str[:100]) if len(line_str) > 100 else line_str)
            elif col.dtype.name == 'bool':
                line.append(str(' | '.join(col.map({True: 'True', False: 'False'}).unique())))
            elif col.dtype.name.startswith('int') \
                    or col.dtype.name.startswith('float') \
                    or col.dtype.name.startswith('date'):
                my_str = 'max=' + str(col.max()) + ' | min=' + str(col.min())
                if col.dtype.name.startswith('date'):
                    my_str += ' | yr mean= ' + str(round(col.dt.year.mean(), 0)).partition('.')[0]
                else:
                    my_str += ' | mean=' + str(round(col.mean(), 2))
                    dominant = col.mode(dropna=True).to_list()[:2]
                    if len(dominant) == 1:
                        dominant = dominant[0]
                    my_str += ' | dominant=' + str(dominant)
                line.append(my_str)
            else:
                line.append('')
            file.append(line)
        df_dd = pd.DataFrame(file, columns=labels)
        if isinstance(report_header, str) and report_header in labels and isinstance(condition, str):
            str_value = "df_dd['{}']{}".format(report_header, condition)
            try:
                df_dd = df_dd.where(eval(str_value)).dropna()
            except(SyntaxError, ValueError):
                pass
        if stylise:
            df_style = df_dd.style.set_table_styles(style)
            _ = df_style.applymap(SyntheticCommons._highlight_null_dom, subset=['%_Null', '%_Dom'])
            _ = df_style.applymap(lambda x: 'color: white' if x > 0.98 else 'color: black', subset=['%_Null', '%_Dom'])
            _ = df_style.applymap(SyntheticCommons._highlight_next, subset=['%_Nxt'])
            _ = df_style.applymap(lambda x: 'color: white' if x < 0.02 else 'color: black', subset=['%_Nxt'])
            _ = df_style.applymap(SyntheticCommons._dtype_color, subset=['dType'])
            _ = df_style.applymap(SyntheticCommons._color_unique, subset=['Unique'])
            _ = df_style.applymap(lambda x: 'color: white' if x < 2 else 'color: black', subset=['Unique'])
            _ = df_style.format({'%_Null': "{:.1%}", '%_Dom': '{:.1%}', '%_Nxt': '{:.1%}'})
            _ = df_style.set_caption('%_Dom: The % most dominant element - %_Nxt: The % next most dominant element')
            _ = df_style.set_properties(subset=[f'Attributes ({len(df.columns)})'],  **{'font-weight': 'bold',
                                                                                        'font-size': "120%"})
            if not inc_next_dom:
                df_style.hide_columns('%_Nxt')
                _ = df_style.set_caption('%_Dom: The % most dominant element')
            return df_style
        if not inc_next_dom:
            df_dd.drop('%_Nxt', axis='columns', inplace=True)
        return df_dd

    @staticmethod
    def _dtype_color(dtype: str):
        """Apply color to types"""
        if str(dtype).startswith('cat'):
            color = '#208a0f'
        elif str(dtype).startswith('int'):
            color = '#0f398a'
        elif str(dtype).startswith('float'):
            color = '#2f0f8a'
        elif str(dtype).startswith('date'):
            color = '#790f8a'
        elif str(dtype).startswith('bool'):
            color = '#08488e'
        elif str(dtype).startswith('str'):
            color = '#761d38'
        else:
            return ''
        return 'color: %s' % color

    @staticmethod
    def _highlight_null_dom(x: str):
        x = float(x)
        if not isinstance(x, float) or x < 0.65:
            return ''
        elif x < 0.85:
            color = '#ffede5'
        elif x < 0.90:
            color = '#fdcdb9'
        elif x < 0.95:
            color = '#fcb499'
        elif x < 0.98:
            color = '#fc9576'
        elif x < 0.99:
            color = '#fb7858'
        elif x < 0.997:
            color = '#f7593f'
        else:
            color = '#ec382b'
        return 'background-color: %s' % color

    @staticmethod
    def _highlight_next(x: str):
        x = float(x)
        if not isinstance(x, float):
            return ''
        elif x < 0.01:
            color = '#ec382b'
        elif x < 0.02:
            color = '#f7593f'
        elif x < 0.03:
            color = '#fb7858'
        elif x < 0.05:
            color = '#fc9576'
        elif x < 0.08:
            color = '#fcb499'
        elif x < 0.12:
            color = '#fdcdb9'
        elif x < 0.18:
            color = '#ffede5'
        else:
            return ''
        return 'background-color: %s' % color

    @staticmethod
    def _color_unique(x: str):
        x = int(x)
        if not isinstance(x, int):
            return ''
        elif x < 2:
            color = '#ec382b'
        elif x < 3:
            color = '#a1cbe2'
        elif x < 5:
            color = '#84cc83'
        elif x < 10:
            color = '#a4da9e'
        elif x < 20:
            color = '#c1e6ba'
        elif x < 50:
            color = '#e5f5e0'
        elif x < 100:
            color = '#f0f9ed'
        else:
            return ''
        return 'background-color: %s' % color


class DataAnalytics(AnalyticsCommons):

    @property
    def relative_freq_map(self):
        return pd.Series(data=self.patterns.relative_freq, index=self.intent.selection, copy=True, dtype=float)

    @property
    def sample_map(self):
        return pd.Series(data=self.stats.sample_distribution, index=self.intent.selection, copy=True, dtype=float)

    @property
    def dominance_map(self):
        return pd.Series(data=self.intent.dominance_weighting, index=self.intent.dominant_values,
                         copy=True, dtype=float)
