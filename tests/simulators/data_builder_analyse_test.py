import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import seaborn as sns
import unittest
import os
import warnings

from ds_behavioral.simulators.data_builder import DataBuilder, DataBuilderTools
from ds_behavioral.config.properties import AbstractPropertiesManager
from ds_behavioral.transition.discovery import DataDiscovery as discover
from ds_behavioral.transition.cleaners import ColumnCleaners as cleaner


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.name='test_build'
        pass

    def tearDown(self):
        _tmp = DataBuilder(self.name).fbpm
        _tmp.remove(_tmp.KEY.manager)
        try:
            os.remove('config_data_builder.yaml')
            os.remove('customer.csv')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        DataBuilder(self.name)

    def test_associate_analysis(self):
        tools = DataBuilderTools()
        df = sns.load_dataset('titanic')
        columns_list = [{'sex': {}}, {'age': {}}, {'survived': {'dtype': 'category'}}]
        analysis = tools.analyse_association(df, columns_list)
        result = tools.associate_analysis(analysis, size=891)
        analysis2 = tools.analyse_association(result, columns_list)
        self.assertEqual(analysis, analysis2)

    def test_titanic(self):
        tools = DataBuilderTools()
        df = sns.load_dataset('titanic')
        column_association = [{'sex': {}},
                              {'age': {'granularity': 16, 'precision': 0}},
                              {'pclass': {'dtype': 'category'}, 'sibsp': {'dtype': 'category'}},
                              {'survived': {'dtype': 'category'}, 'fare': {'granularity': 20, 'precision': 2}},
                              {'embarked': {}}]

        exclude_associate = ['sex.age.sibsp.fare', 'sex.age.sibsp.survived', 'sex.age.pclass.survived.embarked']

        analysis = tools.analyse_association(df, columns_list=column_association, exclude_associate=exclude_associate)
        df_syn = tools.associate_analysis(analysis, size=800)




    @ignore_warnings
    def test_analyse_date(self):
        tools = DataBuilderTools()
        str_dates = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        ts_dates = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        result = tools.analyse_date(str_dates, granularity=6, chunk_size=1)
        control =  [10.0, 30.0, 0.0, 10.0, 10.0, 40.0]
        self.assertEqual(control, result.get('weighting'))
        result = tools.analyse_date(ts_dates, granularity=2, chunk_size=1, date_format='%d-%m-%Y')
        control = [40.0, 60.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual(('09-02-2016','02-12-2017',2), (result.get('lower'), result.get('upper'), result.get('granularity')))
        result = tools.analyse_date(ts_dates, granularity=pd.Timedelta(days=365), chunk_size=1, date_format='%d-%m-%Y')
        control = [40.0, 60.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual(('09-02-2016','02-12-2017',365.0), (result.get('lower'), result.get('upper'), result.get('granularity')))
        print(result)

        result = tools.analyse_date(['12/12/2015'], granularity=2, date_format='%d-%m-%Y')
        print(result)

    def test_analyse_value(self):
        tools = DataBuilderTools()
        dataset = [1,1,1,1,1,1,1,1,1,1]
        result = tools.analyse_number(dataset, granularity=10)
        control = [100.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,1,10), (result.get('lower'), result.get('upper'), result.get('granularity')))
        self.assertEqual([10], result.get('sample'))
        dataset = [1,2,3,4,5,6,7,8,9,10]

        result = tools.analyse_number(dataset, granularity=10)
        control = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10,10), (result.get('lower'), result.get('upper'), result.get('granularity')))

        result = tools.analyse_number(dataset, granularity=1.0)
        control = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10,1.0), (result.get('lower'), result.get('upper'), result.get('granularity')))

        result = tools.analyse_number(dataset, granularity=2.0)
        control = [20.0, 20.0, 20.0, 20.0, 20.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10,2.0), (result.get('lower'), result.get('upper'), result.get('granularity')))

        result = tools.analyse_number(dataset, granularity=3.0)
        control = [30.0, 30.0, 40.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10,3.0), (result.get('lower'), result.get('upper'), result.get('granularity')))

        result = tools.analyse_number(dataset, granularity=1.0, lower=0, upper=10)
        control = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((0,10,1.0), (result.get('lower'), result.get('upper'), result.get('granularity')))

    def test_analyse_value_chunk(self):
        tools = DataBuilderTools()
        values = [1,2,2,1,np.nan,4,2,1,3,9,9,18,21, np.nan]
        result = tools.analyse_number(values, granularity=3, chunk_size=2, replace_zero=0.1)
        control = [[85.71, 0.1, 0.1], [28.57, 28.57, 28.57]]
        self.assertEqual(control, result.get('weighting'))
        control = [14.29, 14.29]
        self.assertEqual(control, result.get('null_values'))
        self.assertEqual([6,6], result.get('sample'))

    def test_analyse_cat(self):
        tools = DataBuilderTools()
        df = sns.load_dataset('titanic')
        df.loc[0, ('sex')] = 'unknown'
        df.loc[600:620, ('sex')] = 'unknown'
        result = tools.analyse_category(df['sex'], chunk_size=2)
        control = {'dtype': 'category', 'selection': ['unknown', 'female', 'male'], 'sample': [446, 445],
                   'weighting': [[0.22, 38.34, 61.43], [4.72, 30.34, 64.94]], 'null_values': [0.0, 0.0]}
        self.assertEqual(control, result)
        df = tools.get_names(size=100, mf_weighting=[60, 40], seed=31, quantity=90.0)
        result = tools.analyse_category(df['gender'])
        control = {'selection': ['M', 'F'], 'weighting': [65.56, 34.44], 'sample': [90], 'dtype': 'category', 'null_values': [10.0]}
        self.assertEqual(control, result)

    @ignore_warnings
    def test_analyse_associate_single(self):
        tools = DataBuilderTools()
        size = 50
        df = tools.get_names(size=size, mf_weighting=[60,40], seed=31, quantity=90.0)
        # category
        columns_list = [{'gender': {}}]
        result = tools.analyse_association(df, columns_list)
        print(result)
        control = {'gender': {'analysis': {'selection': ['F', 'M'], 'weighting': [35.56, 64.44], 'dtype': 'category', 'null_values': [10.0], 'sample': [45]},'associate': 'gender'}}
        self.assertEqual(control, result)
        columns_list = [{'gender': {'chunk_size': 1, 'replace_zero': 0}}]
        result = tools.analyse_association(df, columns_list)
        self.assertEqual(control, result)
        # number
        df['numbers'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'numbers': {'type': 'number', 'granularity': 3}}]
        result = tools.analyse_association(df, columns_list)
        control = {'numbers': {'analysis': {'dtype': 'number', 'granularity': 3,'lower': 9.0, 'null_values': [14.0],
                          'sample': [43],'selection': [(9.0, 324.667),(324.667, 640.333),(640.333, 956.0)],
                          'upper': 956.0,'weighting': [64.0, 0.0, 22.0]},
                   'associate': 'numbers'}}
        self.assertEqual(control, result)
        #dates
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', date_pattern=[1,9,4], size=size, quantity=0.9, seed=31)
        columns_list = [{'dates': {'dtype': 'datetime', 'granularity': 3, 'date_format': '%d-%m-%Y'}}]
        result = tools.analyse_association(df, columns_list)
        control = {
            'dates': {'analysis': {'dtype': 'date', 'granularity': 3, 'lower': '31-12-2000', 'null_values': [10.0],
                                   'selection': [('31-12-2000', '01-09-2006'),
                                                 ('01-09-2006', '01-05-2012'),
                                                 ('01-05-2012', '31-12-2017')],
                                   'upper': '31-12-2017', 'sample': [45], 'weighting': [6.0, 56.0, 28.0]},
                      'associate': 'dates'}}
        self.assertEqual(control, result)

    @ignore_warnings
    def test_multi_analyse_associate(self):
        tools = DataBuilderTools()
        size = 50
        df = tools.get_names(size=size, mf_weighting=[60,40], seed=31, quantity=90.0)
        df['numbers'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', date_pattern=[1,9,4], size=size, quantity=0.9, seed=31)
        columns_list = [{'gender': {}}, {'numbers': {}}]
        result = tools.analyse_association(df, columns_list)
        control = control_01()
        self.assertEqual(control, result)
        df = sns.load_dataset('titanic')
        columns_list = [{'sex': {}}, {'age': {}}, {'survived': {'dtype': 'category'}}]
        result = tools.analyse_association(df, columns_list)
        control = control_02()
        self.assertEqual(control, result)

    def test_levels_analyse_associate(self):
        tools = DataBuilderTools()
        size = 50
        df = tools.get_names(mf_weighting=[60,40], quantity=90.0, seed=31, size=size)
        df['lived'] = tools.get_category(selection=['yes', 'no'], quantity=80.0, seed=31, size=size)
        df['age'] = tools.get_number(from_value=20,to_value=80, weight_pattern=[1,2,5,6,2,1,0.5], seed=31, size=size)
        df['fare'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'gender': {}, 'age':  {}}, {'lived': {}}]
        exclude = ['age.lived']
        result = tools.analyse_association(df, columns_list, exclude)
        control = {}
        self.assertEqual(control, result)

    @ignore_warnings
    def test_date2Value2Date(self):
        tools = DataBuilderTools()
        str_dates = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        ts_dates = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        result = tools.convert_date2value(str_dates)
        control = [736184., 736602., 736003., 736665., 736158., 736547., 736576., 736650., 736374., 736202.]
        self.assertEqual(control, result)
        result = tools.convert_value2date(result, date_format='%d-%m-%Y')
        control = ['08-08-2016', '30-09-2017', '09-02-2016', '02-12-2017', '13-07-2016', '06-08-2017', '04-09-2017', '17-11-2017', '14-02-2017', '26-08-2016']
        self.assertEqual(control, result)

    def get_weights(self, df, columns: list, index: int, weighting: dict):
        col = columns[index]
        weighting.update({col: DataBuilderTools.analyse_category(df[col])})
        if index == len(columns)-1:
            return
        for category in weighting.get(col).get('selection'):
            if weighting.get(col).get('sub_category') is None:
                weighting[col].update({'sub_category': {}})
            weighting.get(col).get('sub_category').update({category: {}})
            sub_category = weighting.get(col).get('sub_category').get(category)
            self.get_weights(df[df[col] == category], columns, index + 1, sub_category)
        return

if __name__ == '__main__':
    unittest.main()


def control_01():
    return {'gender': {'analysis': {'dtype': 'category',
                                    'null_values': [10.0],
                                    'sample': [45],
                                    'selection': ['F', 'M'],
                                    'weighting': [35.56, 64.44]},
                       'sub_category': {'F': {'numbers': {'analysis': {'dtype': 'number',
                                                                       'granularity': 3,
                                                                       'lower': 192.0,
                                                                       'null_values': [0.0],
                                                                       'sample': [16],
                                                                       'selection': [(192.0, 460.667),
                                                                                     (460.667, 729.333),
                                                                                     (729.333, 998.0)],
                                                                       'upper': 998.0,
                                                                       'weighting': [12.5, 37.5, 50.0]}}},
                                        'M': {'numbers': {'analysis': {'dtype': 'number',
                                                                       'granularity': 3,
                                                                       'lower': 14.0,
                                                                       'null_values': [0.0],
                                                                       'sample': [29],
                                                                       'selection': [(14.0, 119.333),
                                                                                     (119.333, 224.667),
                                                                                     (224.667, 330.0)],
                                                                       'upper': 330.0,
                                                                       'weighting': [24.14, 41.38, 34.48]}}}}}}


def control_02():
    return {'sex': {'analysis': {'dtype': 'category',
                                 'null_values': [0.0],
                                 'sample': [891],
                                 'selection': ['male', 'female'],
                                 'weighting': [64.76, 35.24]},
                    'sub_category': {'female': {'age': {'analysis': {'dtype': 'number',
                                                                     'granularity': 3,
                                                                     'lower': 0.75,
                                                                     'null_values': [16.88],
                                                                     'sample': [261],
                                                                     'selection': [(0.75,
                                                                                    21.5),
                                                                                   (21.5,
                                                                                    42.25),
                                                                                   (42.25,
                                                                                    63.0)],
                                                                     'upper': 63.0,
                                                                     'weighting': [26.75,
                                                                                   43.31,
                                                                                   13.06]},
                                                        'sub_category': {(0.75, 21.5): {
                                                            'survived': {'analysis': {'dtype': 'category',
                                                                                      'null_values': [0.0],
                                                                                      'sample': [82],
                                                                                      'selection': [1,
                                                                                                    0],
                                                                                      'weighting': [67.07,
                                                                                                    32.93]}}},
                                                            (21.5, 42.25): {'survived': {
                                                                'analysis': {'dtype': 'category',
                                                                             'null_values': [0.0],
                                                                             'sample': [136],
                                                                             'selection': [1,
                                                                                           0],
                                                                             'weighting': [79.41,
                                                                                           20.59]}}},
                                                            (42.25, 63.0): {'survived': {
                                                                'analysis': {'dtype': 'category',
                                                                             'null_values': [0.0],
                                                                             'sample': [41],
                                                                             'selection': [1,
                                                                                           0],
                                                                             'weighting': [78.05,
                                                                                           21.95]}}}}}},
                                     'male': {'age': {'analysis': {'dtype': 'number',
                                                                   'granularity': 3,
                                                                   'lower': 0.42,
                                                                   'null_values': [21.49],
                                                                   'sample': [453],
                                                                   'selection': [(0.42,
                                                                                  26.947),
                                                                                 (26.947,
                                                                                  53.473),
                                                                                 (53.473,
                                                                                  80.0)],
                                                                   'upper': 80.0,
                                                                   'weighting': [33.28,
                                                                                 38.82,
                                                                                 6.41]},
                                                      'sub_category': {(0.42, 26.947): {
                                                          'survived': {'analysis': {'dtype': 'category',
                                                                                    'null_values': [0.0],
                                                                                    'sample': [191],
                                                                                    'selection': [0,
                                                                                                  1],
                                                                                    'weighting': [79.58,
                                                                                                  20.42]}}},
                                                          (26.947, 53.473): {'survived': {
                                                              'analysis': {'dtype': 'category',
                                                                           'null_values': [0.0],
                                                                           'sample': [224],
                                                                           'selection': [0,
                                                                                         1],
                                                                           'weighting': [78.12,
                                                                                         21.88]}}},
                                                          (53.473, 80.0): {'survived': {
                                                              'analysis': {'dtype': 'category',
                                                                           'null_values': [0.0],
                                                                           'sample': [37],
                                                                           'selection': [0,
                                                                                         1],
                                                                           'weighting': [89.19,
                                                                                         10.81]}}}}}}}}}
