import unittest
import os
import shutil
import numpy as np
import pandas as pd
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentCorrelateTest(unittest.TestCase):

    def setUp(self):
        os.environ['AISTAC_PM_PATH'] = os.path.join('work', 'config')
        os.environ['AISTAC_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['AISTAC_PM_PATH'])
            os.makedirs(os.environ['AISTAC_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    @property
    def tools(self) -> SyntheticIntentModel:
        return SyntheticBuilder.scratch_pad()

    def test_runs(self):
        """Basic smoke test"""
        im = SyntheticBuilder.from_env('tester', default_save=False, default_save_intent=False,
                                       reset_templates=False).intent_model
        self.assertTrue(SyntheticIntentModel, type(im))

    def test_correlate_number(self):
        tools = self.tools
        numbers = [1,2,3,4.0,5,6,7,8,9,0]
        result = tools.correlate_numbers(numbers, label='corr_numbers', precision=0)
        self.assertCountEqual([1,2,3,4,5,6,7,8,9,0], result)
        # Offset
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        result = tools.correlate_numbers(numbers, offset=1, label='corr_numbers', precision=0)
        self.assertEqual([2,3,4,5,6,7,8,9,10,1], result)
        # multiply offset
        numbers = [1, 2, 3, 4]
        result = tools.correlate_numbers(numbers, offset=2, multiply_offset=True, label='corr_numbers', precision=0)
        self.assertEqual([2,4,6,8], result)
        # spread
        numbers = [2] * 1000
        result = tools.correlate_numbers(numbers, spread=5, label='corr_numbers', precision=0)
        self.assertLessEqual(max(result), 4)
        self.assertGreaterEqual(min(result), 0)
        numbers = tools.get_number(99999, size=5000)
        result = tools.correlate_numbers(numbers, spread=5, label='corr_numbers', precision=1)
        self.assertNotEqual(numbers, result)
        self.assertEqual(5000, len(result))
        for index in range(len(result)):
            loss = abs(numbers[index] - result[index])
            self.assertLessEqual(loss, 5)
        numbers = tools.get_number(99999, size=5000)
        result = tools.correlate_numbers(numbers, spread=1, label='corr_numbers', precision=1)
        self.assertNotEqual(numbers, result)
        self.assertEqual(5000, len(result))
        for index in range(len(result)):
            loss = abs(numbers[index] - result[index])
            self.assertLessEqual(loss, 1)

    def test_correlate_number_extras(self):
        tools = self.tools
        # weighting
        numbers = [2] * 1000
        result = tools.correlate_numbers(numbers, spread=5, label='corr_numbers', precision=0, weighting_patten=[0,0,1,1])
        self.assertCountEqual([2,3,4], list(pd.Series(result).value_counts().index))
        result = tools.correlate_numbers(numbers, spread=5, label='corr_numbers', precision=0, weighting_patten=[1,1,0,0])
        self.assertCountEqual([0,1,2], list(pd.Series(result).value_counts().index))
        # fill nan
        numbers = [1,1,2,np.nan,3,1,np.nan,3,5,np.nan,7]
        result = tools.correlate_numbers(numbers, fill_nulls=True, label='corr_numbers', precision=0)
        self.assertEqual([1,1,2,1,3,1,1,3,5,1,7], result)
        numbers = [2] * 1000
        # spread, offset and fillna
        result = tools.correlate_numbers(numbers, offset=2, spread=5, fill_nulls=True, label='corr_numbers', precision=0)
        self.assertCountEqual([2,3,4,5,6], list(pd.Series(result).value_counts().index))
        # min
        numbers = [2] * 100
        result = tools.correlate_numbers(numbers, offset=2, spread=5, min_value=4, label='corr_numbers', precision=0)
        self.assertCountEqual([4, 5, 6], list(pd.Series(result).value_counts().index))
        result = tools.correlate_numbers(numbers, offset=2, spread=5, min_value=6, label='corr_numbers', precision=0)
        self.assertCountEqual([6], list(pd.Series(result).value_counts().index))
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_numbers(numbers, offset=2, spread=5, min_value=7, label='corr_numbers', precision=0)
        self.assertTrue("The min value 7 is greater than the max result value" in str(context.exception))
        # max
        result = tools.correlate_numbers(numbers, offset=2, spread=5, max_value=4, label='corr_numbers', precision=0)
        self.assertCountEqual([2, 3, 4], list(pd.Series(result).value_counts().index))
        result = tools.correlate_numbers(numbers, offset=2, spread=5, max_value=2, label='corr_numbers', precision=0)
        self.assertCountEqual([2], list(pd.Series(result).value_counts().index))
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_numbers(numbers, offset=2, spread=5, max_value=1, label='corr_numbers', precision=0)
        self.assertTrue("The max value 1 is less than the min result value" in str(context.exception))

    def test_correlate_categories(self):
        tools = self.tools
        categories = list("ABCDE")
        correlation = ['A', 'D']
        action = {0: 'F', 1: 'G'}
        result = tools.correlate_categories(categories, correlations=correlation, actions=action, label='letters')
        self.assertEqual(['F', 'B', 'C', 'G', 'E'], result)
        correlation = ['A', 'D']
        action = {0: {'method': 'get_category', 'selection': list("HIJ")}, 1: {'method': 'get_number', 'to_value': 10}}
        result = tools.correlate_categories(categories, correlations=correlation, actions=action, label='letters')
        self.assertIn(result[0], list("HIJ"))
        self.assertTrue(0 <= result[3] < 10)
        categories = tools.get_category(selection=list("ABCDE"), size=5000)
        result = tools.correlate_categories(categories, correlations=correlation, actions=action, label='letters')
        self.assertEqual(5000, len(result))

    def test_correlate_categories_exceptions(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
