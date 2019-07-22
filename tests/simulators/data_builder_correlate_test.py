import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import seaborn as sns
import unittest
import os
import warnings

from ds_behavioral.generator.data_builder import DataBuilder, DataBuilderTools
from ds_discovery.config.abstract_properties import AbstractPropertiesManager


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class FileBuilderTest(unittest.TestCase):

    def setUp(self):
        self.name='test_build'
        pass

    def tearDown(self):
        _tmp = DataBuilder(self.name).fbpm
        _tmp.remove(_tmp.KEY.manager_key)
        try:
            os.remove('config_data_builder.yaml')
            os.remove('customer.csv')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        DataBuilder(self.name)

    def test_correlate_numbers(self):
        tools = DataBuilderTools()
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0]
        control = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 0.5]
        result = tools.correlate_numbers(values, offset=0.5)
        self.assertEqual(control, result)
        result = tools.correlate_numbers(values, spread=1, precision=1, seed=31)
        control = [0.7, 1.7, 3.2, 2.1, 5.7, 5.9, 8.0, 9.4, 8.4, -0.3]
        self.assertEqual(control, result)
        control = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, 0.0]
        result = tools.correlate_numbers(values, quantity=0.9, seed=31)
        self.assertEqual(control, result)
        values = [1.,1.,1.,1.,1.,1.,1.]
        control = [1.244, 1.676, 2.009, 2.405, 1.497, 1.257, 1.026]
        result = tools.correlate_numbers(values, spread=1, seed=31, min_value=1)
        self.assertEqual(control, result)
        values = [1,2,3,4,5,6,7,8,9,0]
        control = [1, 2, 3, 4, 5, 6, 7, 8, None, 0]
        result = tools.correlate_numbers(values, quantity=0.9, seed=31, precision=0)
        self.assertEqual(control, result)
        values = [1,None,5,5,np.nan,5,3,'None',3,3]
        control = [1, 3, 5, 5, 3, 5, 3, 5, 3, 3]
        result = tools.correlate_numbers(values, fill_nulls=True, seed=99)
        self.assertEqual(control, result)
        values = [0,0,0,1,0,2,4,0,1]
        control = [0, 0, 0, -2, 0, 7, 7, 0, -1]
        result = tools.correlate_numbers(values, spread=2, keep_zero=True, seed=99, precision=0)
        self.assertEqual(control, result)
        values = [None, None, 1]
        control = [11,11,11]
        result = tools.correlate_numbers(values, offset=10, fill_nulls=True)
        self.assertEqual(control, result)
        #exception testing
        values = []
        result = tools.correlate_numbers(values, offset=10)
        self.assertEqual(0, len(result))
        values = [None, None]
        result = tools.correlate_numbers(values, offset=10, fill_nulls=True)
        self.assertEqual(values, result)

    @ignore_warnings
    def test_correlate_numbers_max_min(self):
        tools = DataBuilderTools()
        values = [2]
        result = tools.correlate_numbers(values, max_value=1)
        self.assertEqual([1],result)
        result = tools.correlate_numbers(values, min_value=3)
        self.assertEqual([3],result)

    @ignore_warnings
    def test_correlate_date_max_min(self):
        tools = DataBuilderTools()
        dates = ['10/01/2017']
        result = tools.correlate_dates(dates, max_date='05/01/2017')
        self.assertEqual(['05-01-2017T00:00:00'], result)
        result = tools.correlate_dates(dates, min_date='12/01/2017')
        self.assertEqual(['12-01-2017T00:00:00'], result)

    @ignore_warnings
    def test_correlation_dates(self):
        tools = DataBuilderTools()
        dates = ['10/01/2017', '12/01/2017', None, '', 'Fred']
        control = ['10-01-2020', '12-01-2020', None, '', 'Fred']
        result = tools.correlate_dates(dates, lower_spread={'days':2}, upper_spread={'days':2}, offset={'years': 3}, date_format="%d-%m-%Y", seed=99)
        self.assertEqual(control, result)
        control = ['11-01-2017', '13-01-2017', None, '', 'Fred']
        result = tools.correlate_dates(dates, date_format="%d-%m-%Y", seed=99)
        self.assertEqual(control, result)
        dates = DataBuilderTools.get_datetime('01/01/2010', '31/12/2010', date_format="%d-%m-%Y", seed=99, size=5)
        control = ['17-07-2010', '07-07-2010', '06-08-2010', '07-06-2010', '23-02-2010']
        self.assertEqual(control, dates)
        result = tools.correlate_dates(dates, lower_spread={'days':2}, upper_spread={'days':2}, seed=99, date_format="%d-%m-%Y")
        control = ['18-07-2010', '08-07-2010', '06-08-2010', '06-06-2010', '22-02-2010']
        self.assertEqual(control, result)

    @ignore_warnings
    def test_correlation_dates_attributes(self):
        tools = DataBuilderTools()
        rows = 100
        df_staff = pd.DataFrame()
        df_staff['joined'] = tools.get_datetime(start='01/01/2008', until='07/01/2019', date_format='%d-%m-%Y', size=rows)

        def offset_limits():
            diff_list = []
            for index in range(rows):
                c_time = pd.to_datetime(control[index], errors='coerce', infer_datetime_format=True, dayfirst=True)
                r_time = pd.to_datetime(result[index], errors='coerce', infer_datetime_format=True, dayfirst=True)
                diff_list.append(r_time - c_time)
            max_diff = max(diff_list)
            min_diff = min(diff_list)
            mean_diff = np.mean(diff_list)
            return min_diff, mean_diff, max_diff

        control = AbstractPropertiesManager.list_formatter(df_staff['joined'])
        result = tools.correlate_dates(df_staff['joined'], offset={'days': 7})
        min_diff, mean_diff, max_diff = offset_limits()
        self.assertEquals(7, max_diff.days)
        self.assertEquals(6, min_diff.days)

        result = tools.correlate_dates(df_staff['joined'], offset={'days': 7}, lower_spread={'days': 3},
                                       upper_spread={'days': 5})
        min_diff, mean_diff, max_diff = offset_limits()
        self.assertEquals(11, max_diff.days)
        self.assertEquals(4, min_diff.days)

    def test_correlation_category(self):
        tools = DataBuilderTools()
        selection = [ 'F', 'M','U']
        corr= {}
        values = tools.get_category(selection, weight_pattern=[5, 3, 2], size=10)
        result = tools.correlate_categories(values, correlations=selection, actions=corr, value_type='Category')
        self.assertEqual(values, result)

        corr = {0: {'action': 'V'},
                1: {'action': 'get_category', 'kwargs': {'selection' : [0, 1],
                                                         'weight_pattern': [6, 4],
                                                         'seed':101}}}
        values = tools.get_category(selection, weight_pattern=[5,3,2], size=10, seed=101)
        control = ['M', 'F', 'F', 'F', 'F', 'F', 'F', 'M', 'F', 'M']
        self.assertEqual(control, values)
        result = tools.correlate_categories(values, correlations=selection, actions=corr, value_type='C', seed=101)
        control = [0, 'V', 'V', 'V', 'V', 'V', 'V', 0, 'V', 0]
        self.assertEqual(control, result)

        selection = [[1, 5], 6, [7, 9]]
        corr = {0: {'action': 0},
                1: {'action': 1},
                2: {'action': 2}}
        values = tools.get_number(5, 8, weight_pattern=[[1,0], [0,1]], size=10, seed=31)
        control = [5, 5, 6, 5, 5, 7, 7, 7, 6, 6]
        self.assertEqual(control, values)
        result = tools.correlate_categories(values, correlations=selection, actions=corr, value_type='number', seed=101)
        control = [0, 0, 1, 0, 0, 2, 2, 2, 1, 1]
        self.assertEqual(control, result)

        selection = [['11:00', '11:29'], ['11:30', '11:30'], ['11:31', '11:59']]
        corr = {0: {'action': 'Early'},
                1: {'action': 'On-time'},
                2: {'action': 'Late'}}
        values = ['11:23', '11:30', '11:45', '11:02', '11:31']
        result = tools.correlate_categories(values, correlations=selection, actions=corr, value_type='date', seed=101)
        control = ['Early', 'On-time', 'Late', 'Early', 'Late']
        self.assertEqual(control, result)

        selection = [[10, 20],[21,30]]
        corr = {0: {'action': {}},
                1: {'action': 'correlate_numbers', 'kwargs': {'values': {}, 'offset': 100}}}
        values = [11,22]
        result = tools.correlate_categories(values, correlations=selection, actions=corr, value_type='number', seed=101)
        control = [11,122]
        self.assertEqual(control, result)


if __name__ == '__main__':
    unittest.main()
