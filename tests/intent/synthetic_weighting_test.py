import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticWeightingTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
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

    def test_weighting(self):
        size = 1000
        selection = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for i in range(len(selection)):
            pattern = [0]*len(selection)
            pattern[i] = 1
            result = pd.Series(self.tools.get_category(selection=selection, weight_pattern=pattern, size=size))
            self.assertEqual(1, result.nunique())
            self.assertEqual(selection[i], result.unique())



    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
