import unittest
import os
import shutil
import pandas as pd
import numpy as np
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticPipelineTest(unittest.TestCase):

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
    def builder(self) -> SyntheticBuilder:
        return SyntheticBuilder.from_env('tester')

    def test_run_intent_pipeline(self):
        tools = self.builder.intent_model
        tools.get_number(1000, column_name='numbers')
        result = self.builder.pm.report_intent()
        self.assertEqual(['numbers'], result.get('level'))
        self.assertEqual(['0'], result.get('order'))
        self.assertEqual(['get_number'], result.get('intent'))
        self.assertEqual([['range_value=1000', 'column_name=numbers']], result.get('parameters'))
        tools.get_category(selection=['M', 'F'], column_name='gender')
        result = tools.run_intent_pipeline(size=10, columns=['numbers', 'gender', 'jim'])
        self.assertEqual((10, 3), result.shape)
        self.assertEqual((10, 2), result.dropna(axis='columns').shape)
        self.assertCountEqual(['numbers', 'gender'], result.dropna(axis='columns').columns)

    def test_run_synthetic_pipeline(self):
        sb = self.builder
        tools = self.builder.intent_model
        tools.get_number(1000, size=100, column_name='numbers')
        tools.get_category(selection=['M', 'F'], column_name='gender')
        sb.set_outcome()
        sb.run_synthetic_pipeline(size=10)
        result = sb.load_synthetic_canonical()
        print(result)





if __name__ == '__main__':
    unittest.main()
