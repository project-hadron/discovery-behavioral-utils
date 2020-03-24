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
        sample_size = 5000
        result = tools.get_number(10, 1000, label='capital_gain', dominant_values=0, dominant_percent=91.74, precision=2, size=sample_size)
        self.assertEqual(sample_size, len(result))
        result = pd.Series(result)
        count = result.where(result == 0).dropna()
        self.assertTrue(0.91 <= round(len(count)/sample_size, 2) <= 0.93)



if __name__ == '__main__':
    unittest.main()
