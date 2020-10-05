import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith('HADRON'):
                del os.environ[key]

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

    def test_model_columns_headers(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        builder.set_source_uri(uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        result = builder.tools.model_columns(connector_name=builder.CONNECTOR_SOURCE, headers=['survived', 'sex', 'fare'])
        self.assertCountEqual(['survived', 'sex', 'fare'], list(result.columns))

    def test_remove_unwanted_headers(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        builder.set_source_uri(uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        selection = [builder.tools.select2dict(column='survived', condition='==1')]
        result = builder.tools.frame_selection(canonical=builder.CONNECTOR_SOURCE, selection=selection, headers=['survived', 'sex', 'fare'])
        self.assertCountEqual(['survived', 'sex', 'fare'], list(result.columns))
        self.assertEqual(1, result['survived'].min())

    def test_remove_unwanted_rows(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        builder.set_source_uri(uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        selection = [builder.tools.select2dict(column='survived', condition='==1')]
        result = builder.tools.frame_selection(canonical=builder.CONNECTOR_SOURCE, selection=selection)
        self.assertEqual(1, result['survived'].min())

    def test_model_us_zip(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        result = builder.tools.model_us_zip(size=1000, state_code_filter=['NY', 'TX', 'FRED'])
        self.assertCountEqual(['NY', 'TX'], result['StateCode'].value_counts().index.to_list())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
