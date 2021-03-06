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

    def test_relative(self):
        size = 125117 # prime
        freq = [1.,3.,.4]
        result = self.tools._freq_dist_size(relative_freq=freq, size=size)
        self.assertEqual(3, len(result))
        self.assertEqual(size, sum(result))
        result = self.tools._freq_dist_size(relative_freq=freq, size=size, seed=31)
        other = self.tools._freq_dist_size(relative_freq=freq, size=size, seed=31)
        self.assertEqual(size, sum(result))
        self.assertEqual(other, result)

    def test_negtive_dimentions(self):
        selection = ["SydneyCare", "RCP", "Email", "SMS", "AgentAssist"]
        relative_freq=[2,1,2,2,0.1]
        size = 1000
        result = 0
        for i in range(10000):
            select_index = self.tools.get_number(len(selection), relative_freq=relative_freq, size=size, save_intent=dsFalse)
            result += len(select_index)
        print(result/10000)



    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
