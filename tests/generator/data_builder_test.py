import shutil
from collections import Counter

import matplotlib
from ds_behavioral.intent.data_builder_tools import DataBuilderTools
from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_foundation.properties.property_manager import PropertyManager

matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import unittest
import os

from ds_behavioral import DataBuilderComponent

class FileBuilderTest(unittest.TestCase):

    def setUp(self):
        # set environment variables
        os.environ['AISTAC_PM_PATH'] = os.path.join(os.environ['PWD'], 'work', 'config')
        self.name = 'TestBuilder'
        try:
            shutil.copytree('../simulators/data', os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass
        props = PropertyManager().get_all()
        for key in props.keys():
            PropertyManager().remove(key)

    def test_runs(self):
        """Basic smoke test"""
        DataBuilderComponent.from_env(self.name)

    def test_tools(self):
        """test we can get tools"""
        fb = DataBuilderComponent.from_env(self.name)
        self.assertEqual(fb.tool_dir, DataBuilderTools.__dir__())

    def test_from(self):
        tools = DataBuilderTools
        values = list('ABCDEFG')
        result = tools.get_from(values=values, size=100)
        self.assertEqual(100, len(result))
        for v in values:
            self.assertIn(v, result)
        result = tools.get_from(values=values, sample_size=3, size=100)
        self.assertEqual(100, len(result))
        for v in list('ABC'):
            self.assertIn(v, result)
        for v in list('DEFG'):
            self.assertNotIn(v, result)

    def test_file_column(self):
        tools = DataBuilderTools
        df = pd.DataFrame()
        df['cat'] = tools.get_category(list('MFU'), size=10, seed=31)
        df['values'] = tools.get_number(10, size=10, seed=31)
        df.to_csv("test_df.csv", sep=',',  index=False)
        connector_contract = ConnectorContract(uri="test_df.csv",
                                               module_name='ds_foundation.handlers.python_handlers',
                                               handler='PythonSourceHandler')
        result = tools.get_file_column('cat', connector_contract, size=3, seed=31)
        self.assertEqual((3,1), result.shape)

    def test_category(self):
        tools = DataBuilderTools
        gender = list('MFTU')
        int_values = [1,2,3,4]
        float_values = [0.1,0.2,0.3,0.4]
        for selection in [gender, int_values, float_values]:
            result = tools.get_category(selection=selection, size=1000)
            for i in result:
                self.assertIn(i, selection)
            # print(pd.Series(result).value_counts())
            for c in pd.Series(result).value_counts():
                self.assertTrue(220 < c < 280)

    def test_category_weight_pattern(self):
        tools = DataBuilderTools
        selection = list('MFTU')
        weight_pattern=[5,3,0.1,0.9]
        result = tools.get_category(selection=selection, weight_pattern=weight_pattern, size=1000)
        print(pd.Series(result).value_counts())

    def test_category_at_most(self):
        tools = DataBuilderTools
        selection = list('MFTU')
        result = tools.get_category(selection=selection, at_most=1, size=4)
        self.assertEqual(1, max(pd.Series(result).value_counts()))
        result = tools.get_category(selection=selection, at_most=3, size=10)
        self.assertEqual(3, max(pd.Series(result).value_counts()))
        weight_pattern=[5,3,0.1,0.9]
        result = tools.get_category(selection=selection, weight_pattern=weight_pattern, at_most=10, size=40)
        print(pd.Series(result).value_counts())

    def test_custom(self):
        tools = DataBuilderTools
        result = tools.get_custom('round(np.random.random(),3)', seed=101)
        self.assertEqual([0.598], result)
        result = tools.get_custom("DataBuilderTools.unique_str_tokens(length, ntokens)", length=3, ntokens=4, seed=101, size=3)
        control = [['aZo', 'YYV', 'iTi', 'cRj'], ['hjt', 'xRm', 'CvS', 'BgX'], ['iKW', 'Jzn', 'oMy', 'fbz']]
        for i in range(len(control)):
            self.assertEqual(set(control[i]), set(result[i]))
        control = [0.567]
        result = tools.get_custom('round(np.random.normal(loc=loc, scale=scale),3)', seed=101, loc=0.4, scale=0.1)
        self.assertEqual(control, result)
        control = [17.081]
        result = tools.get_custom('round(np.random.normal(loc=loc, scale=scale),3)', seed=101, loc=0.4, scale=10)
        self.assertEqual(control, result)

    def test_quantity_number(self):
        tools = DataBuilderTools
        result = tools.get_number(100, size=10, quantity=1, seed=101)
        control = [60, 68, 30, 73, 58, 81, 17, 31, 36, 49]
        self.assertEqual(control, result)
        result = tools.get_number(100, size=10, quantity=0.9, seed=98)
        control = [67, 49, 83, 3, 81, 57, 30, 5, None, 1]
        self.assertEqual(control, result)
        result = tools.get_number(100, size=10, quantity=0.5, seed=98)
        control = [None, 49, None, 3, None, None, 30, 5, None, 1]
        self.assertEqual(control, result)
        result = tools.get_number(100, size=10, quantity=0.1, seed = 31)
        control = [None, 24, None, 45, None, None, None, None, None, None]
        self.assertEqual(control, result)
        result = tools.get_number(100, size=10, quantity=0)
        control = [None, None, None, None, None, None, None, None, None, None]
        self.assertEqual(control, result)

    def test_quality_cat(self):
        tools = DataBuilderTools
        cats = ['B', 'L', 'W', 'C', 'V', 'I', 'A', 'K', 'N', 'F', 'R', 'X', 'J', 'M', 'T']
        result = tools.get_category(cats, size=10, quantity=1, seed=99)
        control = list('NKNAWLBWCF')
        self.assertEqual(''.join(control), ''.join(result))
        result = tools.get_category(cats, size=10, quantity=0.5, seed=99)
        control = ['KABCF']
        self.assertEqual(''.join(control), ''.join(result))
        result = tools.get_category(cats, size=10, quantity=0.1, seed=101)
        control = ['C']
        self.assertEqual(''.join(control), ''.join(result))
        result = tools.get_category(cats, size=10, quantity=0)
        control = ['' '' '' '' '' '' '' '' '' '']
        self.assertEqual(''.join(control), ''.join(result))

    def test_distribution(self):
        tools = DataBuilderTools
        result = tools.get_distribution(method='beta', seed=101, a=5, b=2)
        self.assertEqual([0.745], result)

    def test_interval(self):
        tools = DataBuilderTools
        intervals = [(0,10),(10,20),(20,30),(30,40)]
        weighting = [1,1,1,1]
        result = tools.get_intervals(intervals, weight_pattern=weighting, size=10, seed=31)
        self.assertTrue(all(0 <= x <= 40 for x in result))
        self.assertEqual(10, len(result))

    def test_get_one_hot(self):
        tools = DataBuilderTools
        selection = list('ABCDE')
        result = tools.get_one_hot(selection=selection, size=10, weight_pattern=[0,0,1,1,1], quantity=0.5)
        print(result)


    def test_int(self):
        tools = DataBuilderTools
        result = tools.get_number(1,1.0, size=3)
        control = [1.0, 1.0, 1.0]
        self.assertEqual(control, result)
        result = tools.get_number(1,1, size=3)
        control = [1, 1, 1]
        self.assertEqual(control, result)
        result = tools.get_number(from_value=20)
        self.assertEqual(len(result), 1)
        self.assertTrue(0 <= result[0] <= 20)
        result = tools.get_number(from_value=100, quantity=0.5, size=10)
        self.assertIn(None, result)
        pattern = [0,1]
        result = tools.get_number(from_value=1, to_value=100, weight_pattern=pattern, size=1000)
        self.assertTrue(all(isinstance(x, int) for x in result))
        result = tools.get_number(from_value=1, to_value=100, weight_pattern=pattern, size=20)
        self.assertTrue(min(result) >= 50)

    def test_number_dominant(self):
        tools = DataBuilderTools
        result = tools.get_number(from_value=20, dominant_values=[0, 1], dominant_percent=0.6, dominance_weighting=[7, 3], size=10)


    def test_float(self):
        tools = DataBuilderTools
        result = tools.get_number(from_value=1.0, precision=3, seed=101)
        self.assertEqual([0.598], result)
        pattern = [1,1]
        control = [0.299, 0.338, 0.15, 0.365, 0.289, 0.905, 0.584, 0.656, 0.679, 0.746]
        result = tools.get_number(from_value=0, to_value=1.0, weight_pattern=pattern, size=10, seed=101)
        self.assertEqual(control, result)
        result = tools.get_number(from_value=1000, to_value=2000, size=10, currency='£', precision=0, seed=31)
        control = ['£1,859','£1,249','£1,039','£1,458','£1,729','£1,944','£1,385', '£1,547','£1,408', '£1,251']
        self.assertEqual(control, result)
        result = tools.get_number(from_value=1000, to_value=2000, size=10, currency='$', precision=2, seed=31)
        control = ['$1,858.00', '$1,248.00', '$1,038.00', '$1,458.00', '$1,728.00', '$1,944.00', '$1,384.00', '$1,546.00', '$1,407.00', '$1,250.00']
        self.assertEqual(control, result)

    def test_number_offset(self):
        tools = DataBuilderTools
        result = tools.get_number(20, seed=101, size=10, offset=1000)
        control = [12000, 14000, 6000, 15000, 12000, 16000, 3000, 6000, 7000, 10000]
        self.assertEqual(control, result)

    def test_unique_num(self):
        tools = DataBuilderTools
        result = tools.unique_numbers(4, 100, size=5, seed=101)
        self.assertEqual([56, 15, 19, 25, 44], result)
        result = tools.unique_numbers(4, 100090, size=10000,)
        self.assertEqual(1, max(Counter(result).values()))
        result = tools.unique_numbers(10, at_most=3, size=30)
        self.assertEqual(3, max(Counter(result).values()))

    def test_bool(self):
        tools = DataBuilderTools
        count_weight = 0
        count_probability = 0
        count_default = 0
        for i in range(100):
            self.assertEqual([True], tools.get_category([True,False], weight_pattern=[1, 0]))
            self.assertEqual([False], tools.get_category([True,False], weight_pattern=[0, 1]))
            if tools.get_category([True, False], weight_pattern=[5, 5], size=1, seed=i)[0]:
                count_weight += 1
            if tools.get_category([True, False], weight_pattern=[0.5, 0.5], size=1, seed=i)[0]:
                count_probability += 1
            if tools.get_category([True, False], seed=i)[0]:
                count_default += 1
        self.assertEqual(53, count_default)
        self.assertEqual(53, count_probability)
        self.assertEqual(53, count_weight)

    def test_quantity_nulls(self):
        tools = DataBuilderTools
        test = [1,5,3]
        result = tools._set_quantity(test, quantity=0)
        control = [None, None, None]
        self.assertEqual(control, result)
        test = ['a','b','c']
        result = tools._set_quantity(test, quantity=0)
        control = ['', '', '']
        self.assertEqual(control, result)
        test = [0.1,0.5,0.7]
        result = tools._set_quantity(test, quantity=0)
        control = [np.nan, np.nan, np.nan]
        self.assertEqual(control, result)
        test = ['a',2,0.7]
        result = tools._set_quantity(test, quantity=0)
        control = ['', None, np.nan]
        self.assertEqual(control, result)

    def test_string_pattern(self):
        tools = DataBuilderTools
        pattern = 'cd-Uls:ddp'
        result = tools.get_string_pattern(pattern, size=2, seed=101)
        control = ['a3-Os :22,', 'h9-Tz :61{']
        self.assertEqual(control, result)
        pattern = 'abcdefg'
        result = tools.get_string_pattern(pattern, size=2, seed=101)
        control = ['a3', 'h9']
        self.assertEqual(control, result)
        result = tools.get_string_pattern(pattern, choice_only=True, size=2, seed=101)
        self.assertEqual(control, result)
        result = tools.get_string_pattern(pattern, choice_only=False, size=2, seed=101)
        control = ['aba3efg', 'abh9efg']
        self.assertEqual(control, result)

    def test_string_pattern_choices(self):
        tools = DataBuilderTools
        choices = {'?': list('ABC'), '|': ['Code','Ref', 'id'], '#': list('1234')}
        pattern = 'Hi ?, your reference is: |#'
        result = tools.get_string_pattern(pattern, choices=choices, size=3, seed=101)
        control = ['Aid3', 'BRef3', 'BRef2']
        self.assertEqual(control, result)
        result = tools.get_string_pattern(pattern, choices=choices, choice_only=False, size=3, seed=101)
        control = ['Hi A, your reference is: id3', 'Hi B, your reference is: Ref3', 'Hi B, your reference is: Ref2']
        self.assertEqual(control, result)
        choices = {'#': [str(x) for x in range(10, 15)]}
        result = DataBuilderTools.get_string_pattern("Number #", choices=choices, choice_only=False, size=5, seed=31)
        control = ['Number 13', 'Number 14', 'Number 11', 'Number 11', 'Number 12']
        self.assertEqual(control, result)
        choices = {'#': list('abcdef') + list('0123456789'), '-': ['-']}
        result = tools.get_string_pattern('########-####-####-####-############', choices=choices, seed=31)
        control = ['15f0682d-6183-df43-e5bd-bcd24c4ecee3']
        self.assertEqual(control, result)

    def test_tagged_pattern(self):
        tools = DataBuilderTools
        text = "My name is <name>, how do you do"
        tags = {"<name>": {'action': 'get_category', 'kwargs': {'selection': ['fred', 'jim']}}}
        result = tools.get_tagged_pattern(text, tags=tags)
        print(result)

    def test_unique_date(self):
        tools = DataBuilderTools
        start = "01/01/2016"
        end = "31/12/2018"
        result = tools.unique_date_seq(start, end, date_format="%d-%m-%Y", size=100)
        self.assertEqual(100, len(set(result)))
        start = "01/01/2018 00:00"
        end = "01/01/2018 00:00"
        with self.assertRaises(InterruptedError) as context:
            tools.unique_date_seq(start, end, size=61)
        self.assertTrue('Unique Date Sequence stopped' in str(context.exception))
        start = "01/12/2018"
        end = "31/12/2018"
        result = tools.unique_date_seq(start, end, size=1000)
        self.assertEqual(1000, len(set(result)))

    def test_datetime(self):
        tools = DataBuilderTools
        start = "11/11/1964"
        until = "01/01/2018"
        control = ['15-08-1996', '28-10-1987', '20-09-1972']
        result = tools.get_datetime(start, until, date_format="%d-%m-%Y", size=3, seed=101)
        self.assertEqual(control, result)
        control = ['13-02-1974', '29-12-1976', '02-08-1982']
        result = tools.get_datetime(start, until, date_format="%d-%m-%Y", size=3, date_pattern=[1], seed=101)
        self.assertEqual(control, result)

    def test_date_pattern_exceptions(self):
        tools = DataBuilderTools
        valid = '23-12-2016'
        for d in ['', 'text', None]:
            with self.assertRaises(ValueError) as context:
                tools.get_datetime(d, '')
            self.assertTrue("The start or until parameters cannot be" in str(context.exception))
            with self.assertRaises(ValueError) as context:
                tools.get_datetime(valid, d)
            self.assertTrue("The start or until parameters cannot be" in str(context.exception))

    def test_date_year_pattern(self):
        tools = DataBuilderTools
        start = '01/01/2010'
        until = '31/12/2019'
        result = tools.get_datetime(start=start, until=until, default='28/01/2017', year_pattern=[1], size=100, seed=101)
        test = [0]*10
        for d in result:
            self.assertEqual(28, d.day)
            self.assertEqual(1, d.month)
            test[d.year-2010] += 1
        control = [8, 14, 11, 7, 6, 8, 12, 10, 9, 15]
        self.assertEqual(control, test)
        for i in range(10):
            pattern = [0]*10
            pattern[i] = 1
            result = tools.get_datetime(start=start, until=until, default='28/01/2017', year_pattern=pattern, size=10, seed=101)
            test = [0]*10
            for d in result:
                self.assertEqual(28, d.day)
                self.assertEqual(1, d.month)
                test[d.year - 2010] += 1
            self.assertEqual(10, test[i])
            self.assertEqual(10, sum(test))
        pattern = [0]
        control = [pd.NaT, pd.NaT, pd.NaT]
        result = tools.get_datetime(start=start, until=until, year_pattern=pattern, size=3)
        self.assertEqual(control, result)

    def test_date_month_pattern(self):
        tools = DataBuilderTools
        start = '01/01/2017'
        until = '01/01/2018'
        result = tools.get_datetime(start=start, until=until, default='31/01/2016', month_pattern=[1], size=100, seed=101)
        test = [0]*12
        for d in result:
            self.assertEqual(d.daysinmonth, d.day)
            self.assertEqual(2016, d.year)
            test[d.month-1] += 1
        control = [10, 7, 10, 10, 8, 5, 4, 9, 12, 10, 9, 6]
        self.assertEqual(control, test)
        for i in range(12):
            pattern = [0]*12
            pattern[i] = 1
            result = tools.get_datetime(start=start, until=until, default='31/01/2016', month_pattern=pattern, size=10, seed=101)
            test = [0]*12
            for d in result:
                self.assertEqual(d.daysinmonth, d.day)
                self.assertEqual(2016, d.year)
                test[d.month - 1] += 1
            self.assertEqual(10, test[i])
            self.assertEqual(10, sum(test))
        pattern = [0]
        control = [pd.NaT, pd.NaT, pd.NaT]
        result = tools.get_datetime(start=start, until=until, year_pattern=pattern, size=3)
        self.assertEqual(control, result)
        result = tools.get_datetime('01/01/2017', '31/12/2017', size=100, date_format='%d/%m/%Y', ordered=True, month_pattern=[4,0,1,4])

    def test_date_week_pattern(self):
        tools = DataBuilderTools
        start = '01/01/2017'
        until = '31/01/2017'
        start_days = ['02', '03', '04', '05', '06', '07', '01']
        end_days = ['30', '31', '25', '26', '27', '28', '29']
        for i in range(7):
            # end of range
            pattern = [0] * i + [1] + [0] * (6 - i)
            control = ['{}-01-2017'.format(end_days[i])]
            result = tools.get_datetime(start=start, until=until, default='31/01/2017', weekday_pattern=pattern,
                                      date_format='%d-%m-%Y', seed=11)
            self.assertEqual(control, result)
            # start of range
            control = ['{}-01-2017'.format(start_days[i])]
            result = tools.get_datetime(start=start, until=until, default='01/01/2017', weekday_pattern=pattern,
                                      date_format='%d-%m-%Y', seed=11)
            self.assertEqual(control, result)
        start = '14/01/2019'
        until = '20/01/2019'
        pattern = [1]*5 + [0]*2
        result = tools.get_datetime(start=start, until=until, default='17/01/2019', weekday_pattern=pattern,
                                  date_format='%d-%m-%Y', size=100)
        control = [0]*7
        for r in result:
            day = (str(r).split(sep='-'))[0]
            control[int(day)-14] +=1
        self.assertEqual(0, control[5])
        self.assertEqual(0, control[6])
        for i in range(5):
            self.assertTrue(control[i] > 0)
        # test NaT
        start = '01/01/2017'
        until = '02/01/2017'
        pattern = [1] + [0] * 6
        result = tools.get_datetime(start=start, until=until, weekday_pattern=pattern, date_format='%d-%m-%Y', seed=11)
        self.assertEqual(['NaT'], result)

    def test_date_hour_pattern(self):
        tools = DataBuilderTools
        start = '01/01/2018 00:00'
        until = '01/02/2018 00:00'
        result = tools.get_datetime(start=start, until=until, default='01/02/2018 12:23:00', hour_pattern=[1], size=100, seed=101)
        test = [0]*24
        for d in result:
            self.assertEqual(1, d.day)
            self.assertEqual(2, d.month)
            self.assertEqual(2018, d.year)
            self.assertEqual(23, d.minute)
            test[d.hour] += 1
        control = [6, 9, 6, 4, 5, 3, 6, 2, 5, 4, 3, 0, 7, 6, 6, 4, 2, 5, 1, 2, 5, 4, 5, 0]
        self.assertEqual(control, test)
        for i in range(24):
            pattern = [0]*24
            pattern[i] = 1
            result = tools.get_datetime(start=start, until=until, default='01/02/2018 12:23:00', hour_pattern=pattern, size=10, seed=101)
            test = [0]*24
            for d in result:
                # self.assertEqual(1, d.day)
                # self.assertEqual(2, d.month)
                # self.assertEqual(2018, d.year)
                # self.assertEqual(23, d.minute)
                test[d.hour] += 1
            print(test)
            # self.assertEqual(10, test[i])
            # self.assertEqual(10, sum(test))

    def test_norm_weight(self):
        selection = DataBuilderTools.get_string_pattern("UUUU", size=20)
        pattern = [1, 3, 4, 5, 6]
        result = DataBuilderTools._normailse_weights(pattern, length=len(selection))
        self.assertEqual(len(selection), len(result))

    def test_category_replace(self):
        tools = DataBuilderTools
        selection = [1,2,3,4,5,6,7,8,9,0]
        control = [9, 3, 1, 5, 8, 0, 4, 6, 5, 3]
        result = tools.get_category(selection, size=10, seed=31)
        self.assertEqual(control, result)
        result = tools.get_category(selection, size=10, at_most=1)
        for i in result:
            self.assertIn(i, selection)

    def test_profile(self):
        tools = DataBuilderTools
        result = tools.get_profiles(size=10, dominance=[0, 1])
        self.assertEqual(['F'], result['gender'].unique())
        result = tools.get_profiles(size=10, dominance=[1, 0])
        self.assertEqual(['M'], result['gender'].unique())

        result = tools.get_profiles(size=6000, dominance=[1, 1])
        counter = result.groupby('gender').nunique(dropna=True).drop('gender', axis=1)
        self.assertTrue(counter.iloc[0,0] > counter.iloc[0,1])
        self.assertTrue(counter.iloc[1,0] > counter.iloc[1,1])

    def test_local_get_number(self):
        pattern = [0.2, 1, 3, 4, 6, 7, 3, 2, 0.1]

        start = 50.1
        end = 1000
        sample_size = 10

        result = DataBuilderTools.get_number(from_value=start, to_value=end, weight_pattern=pattern, size=sample_size)
        print(result)



if __name__ == '__main__':
    unittest.main()
