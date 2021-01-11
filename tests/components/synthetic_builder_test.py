import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticBuilderTest(unittest.TestCase):

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

    def test_runs(self):
        """Basic smoke test"""
        self.assertEqual(SyntheticBuilder, type(SyntheticBuilder.from_env('tester', has_contract=False)))

    def test_run_synthetic_pipeline_seed(self):
        builder = SyntheticBuilder.from_env('tester', has_contract=False)
        builder.set_persist()
        tools: SyntheticIntentModel = builder.tools
        _ = tools.get_category(selection=['M', 'F'], relative_freq=[4, 3], column_name='gender')
        _ = tools.get_number(from_value=18, to_value=80, column_name='age')
        builder.run_synthetic_pipeline(size=1000, seed=23)
        df = builder.load_synthetic_canonical()
        dist = df['gender'].value_counts().values
        mean = df['age'].mean()
        builder.run_synthetic_pipeline(size=1000, seed=23)
        df = builder.load_synthetic_canonical()
        self.assertCountEqual(dist, df['gender'].value_counts().values)
        self.assertEqual(mean, df['age'].mean())

    def test_report_attr_desc(self):
        builder = SyntheticBuilder.from_env('tester', has_contract=False)
        builder.set_persist()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['gender'] = tools.get_category(selection=['M', 'F'], relative_freq=[4, 3], column_name='gender')
        builder.add_column_description(column_name='gender', description='The gender of the persona')
        df['age'] = tools.get_number(from_value=18, to_value=80, column_name='age')
        builder.add_column_description(column_name='age', description='A one time age field')
        result = builder.report_attr_desc(df, stylise=False)
        self.assertEqual(['A one time age field', 'The gender of the persona'], result.iloc[:,2].to_list())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
