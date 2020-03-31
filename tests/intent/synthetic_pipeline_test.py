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
        return SyntheticBuilder.from_env('tester', default_save=False, default_save_intent=True, reset_templates=False)

    def test_run_pipeline(self):
        tools = self.builder.intent_model
        tools.get_number(1000, size=100, column_name='numbers')
        result = self.builder.pm.report_intent()
        self.assertEqual(['numbers'], result.get('level'))
        self.assertEqual(['0'], result.get('order'))
        self.assertEqual(['get_number'], result.get('intent'))
        self.assertEqual([['range_value=1000', 'column_name=numbers']], result.get('parameters'))
        result = tools.run_intent_pipeline(size=10)
        self.assertEqual((10, 1), result.shape)
        self.assertEqual(['numbers'], result.columns)
        print(result['numbers'])




if __name__ == '__main__':
    unittest.main()
