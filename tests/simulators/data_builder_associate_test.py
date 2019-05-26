import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import seaborn as sns
import unittest
import os
import warnings

from ds_behavioral.generator.data_builder import DataBuilder, DataBuilderTools
from ds_discovery.config.properties import AbstractPropertiesManager
from ds_discovery.transition.discovery import DataDiscovery as discover
from ds_discovery.transition.cleaners import ColumnCleaners as cleaner


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

    def test_associate_custom(self):
        tools = DataBuilderTools()
        df = pd.DataFrame()
        df['cat'] = tools.get_category(list('MFU'), size=10, seed=31)
        df['values'] = tools.get_number(10, size=10, seed=31)
        control = [2.0,np.nan,0.0,np.nan,2.0,np.nan,np.nan,np.nan,5.0,3.0]
        result = tools.associate_custom(df=df, code_str="df.loc[df['cat'] == 'U', 'values'] = new_value", use_exec=True, new_value=None)
        # self.assertEqual(control,list(result['values']))
        df['values'] = tools.get_number(10, size=10, seed=31)
        result = tools.associate_custom(df=df, code_str="df['values'] = np.where(df['cat'] == 'U', None, df['values'])", use_exec=True, new_value=None)
        control = [None, 2, 0, 4, None, None, 3, 5, 4, 2]
        self.assertEqual(control,list(result['values']))
        self.assertEqual(control,list(df['values']))

        df_staff = pd.DataFrame()
        df_staff['sid'] = [1000]
        result = tools.associate_custom(df=df_staff, code_str="df[header].apply(lambda x: f'CU_{x}')", header='sid')
        self.assertEqual(['CU_1000'], list(result))

    def test_associate_dataset(self):
        tools = DataBuilderTools()
        dataset = ['M', 'F', 'M', 'M', 'U', 'F']
        associations = [{'expect': 'c', 'value': 'M'},
                        {'expect': 'c', 'value': 'F'}]
        actions = {0: {'action': 'Male'},
                   1: {'action': 'Female'}}
        result = tools. associate_dataset(dataset, associations=associations, actions=actions, default_header='_default')
        control = ['Male', 'Female', 'Male', 'Male', 'U', 'Female']
        self.assertEqual(control, result)

        dataset = pd.DataFrame()
        dataset['gender'] = ['M', 'F', 'M', 'M', 'U', 'F', 'U']
        dataset['age'] = [20, 22, 18, 43, 34, 57, 22]
        associations = [{'age': {'expect': 'n', 'value': [24, 100]},
                         'gender': {'expect': 'category','value': ['M']}},
                        {'age': {'expect': 'n', 'value': [0, 23]}},
                        {'age': {'expect': 'n', 'value': [24, 100]},
                         'gender': {'expect': 'category', 'value': ['F']}},
                        ]
        actions = {0: {'action': 'Dad'},
                   1: {'action': 'correlate_numbers', 'kwargs': {'values': {'_header': 'age'}, 'offset' : 100}},
                   2: {'action': {'_header': 'age'}}}

        result = tools.associate_dataset(dataset, associations=associations, actions=actions, default_value='Unknown')
        control = [120, 122, 118, 'Dad', 'Unknown', 57, 122]
        self.assertEqual(control, result)

        titanic = sns.load_dataset('titanic')
        associations = [{'sex': {'expect': 'category', 'value': ['male']},
                         'survived': {'expect': 'number', 'value': 0}},
                        {'sex': {'expect': 'category', 'value': ['male']},
                         'survived': {'expect': 'number', 'value': 1}},
                        {'sex': {'expect': 'category', 'value': ['female']},
                         'survived': {'expect': 'number', 'value': 0}},
                        {'sex': {'expect': 'category', 'value': ['female']},
                         'survived': {'expect': 'number', 'value': 1}}]

        actions = {0: {'action': 'correlate_numbers', 'kwargs': {'values': {'_header': 'age'}, 'fill_nulls': True}},
                   1: {'action': 'correlate_numbers', 'kwargs': {'values': {'_header': 'age'}, 'fill_nulls': True}},
                   2: {'action': 'correlate_numbers', 'kwargs': {'values': {'_header': 'age'}, 'fill_nulls': True}},
                   3: {'action': 'correlate_numbers', 'kwargs': {'values': {'_header': 'age'}, 'fill_nulls': True}},
                   }

        result = tools.associate_dataset(titanic, associations=associations, actions=actions, default_value=99)
        control = [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0, 14.0]
        self.assertEqual(control, result[:10])



    @ignore_warnings
    def test_associate_timeseries(self):
        tools = DataBuilderTools()
        rows = 100
        df_staff = pd.DataFrame()
        df_staff['sid'] = tools.unique_identifiers(from_value=10000000, to_value=99999999, size=rows)
        df_staff['staff_type'] = tools.get_category(selection=['contractor', 'part-time', 'full-time'],
                                                    weight_pattern=[1, 3, 6], size=rows)
        df_staff['joined'] = tools.get_datetime(start='01/01/2008', until='07/01/2019', date_format='%d-%m-%Y',
                                                size=rows)

        associations = [{'joined': {'expect': 'date', 'value': ['01/01/2000', '31/12/2013']},
                         'staff_type': {'expect': 'category', 'value': ['full-time', 'part-time']}},
                        {'joined': {'expect': 'date', 'value': ['31/12/2013', '31/12/2100']},
                         'staff_type': {'expect': 'category', 'value': ['full-time', 'part-time']}}]

        actions = {0: {'action': 'get_datetime', 'kwargs': {'start': "05/01/2014", 'until': "16/01/2014"}},
                   1: {'action': 'correlate_dates',
                       'kwargs': {'dates': {'_header': 'joined'}, 'offset': {'days': 9}, 'lower_spread': 4}}}

        df_staff['registered'] = tools.associate_dataset(df_staff, associations=associations, actions=actions,
                                                         default_value=None)
        df_staff = cleaner.to_date_type(df_staff, headers=['registered', 'joined'])
        # TODO finish this


if __name__ == '__main__':
    unittest.main()
