import unittest
import os
import shutil
import pandas as pd
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from ds_behavioral import SyntheticBuilder
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentGetTest(unittest.TestCase):

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

    def test_get_number(self):
        tools = self.tools
        sample_size = 100
        # from to int
        result = tools.get_number(10, 1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 10)
        self.assertLess(max(result), 1000)
        result = tools.get_number(10, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0)
        self.assertLess(max(result), 10)
        # from to float
        result = tools.get_number(0.1, 0.9, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0.1)
        self.assertLess(max(result), 0.9)
        result = tools.get_number(1.0, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0)
        self.assertLess(max(result), 1.0)
        # from
        result = tools.get_number(10, 1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 10)
        self.assertLess(max(result), 1000)
        # set both
        result = tools.get_number(range_value=10, to_value=1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 10)
        self.assertLess(max(result), 1000)
        # set to_value
        result = tools.get_number(to_value=1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0)
        self.assertLess(max(result), 1000)

    def test_get_number_at_most(self):
        tools = self.tools
        sample_size = 100
        result = tools.get_number(10.0, 1000.0, precision=2, at_most=1, size=sample_size)
        result = pd.Series(result)
        self.assertEqual(sample_size, pd.Series(result).nunique())
        self.assertGreaterEqual(pd.Series(result).min(), 10)
        self.assertLessEqual(pd.Series(result).max(), 1000)

    def test_get_number_dominant(self):
        tools = self.tools
        sample_size = 5000
        result = tools.get_number(10, 1000, dominant_values=0, dominant_percent=91.74, precision=2, size=sample_size)
        self.assertEqual(sample_size, len(result))
        result = pd.Series(result)
        count = result.where(result == 0).dropna()
        self.assertTrue(0.91 <= round(len(count)/sample_size, 2) <= 0.93)

    def test_get_datetime_at_most(self):
        tools = self.tools
        sample_size = 3
        result = tools.get_datetime('2018/01/01', '2019/01/01', at_most=1, size=sample_size)
        print(result)




if __name__ == '__main__':
    unittest.main()
