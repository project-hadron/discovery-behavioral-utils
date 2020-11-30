import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from aistac.properties.property_manager import PropertyManager

from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel


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
        self.assertCountEqual(['StateAbbrev', 'Zipcode', 'City', 'State', 'StateCode', 'Phone'], result.columns.to_list())
        self.assertEqual(1000, result.shape[0])

    def test_model_us_person(self):
        builder = SyntheticBuilder.from_memory(default_save_intent=False)
        result = builder.tools.model_person(size=1000)
        self.assertCountEqual(['given_name', 'gender', 'family_name', 'initials', 'email'], result.columns.to_list())
        self.assertEqual(1000, result.shape[0])

    def test_model_iterator(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri('titanic', uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        # do nothing
        result = tools.model_iterator(canonical='titanic')
        self.assertEqual(builder.load_canonical('titanic').shape, result.shape)
        # add marker
        result = tools.model_iterator(canonical='titanic', marker_col='marker')
        self.assertEqual(builder.load_canonical('titanic').shape[1]+1, result.shape[1])
        # with selection
        selection = [tools.select2dict(column='survived', condition="==1")]
        control = tools.frame_selection(canonical='titanic', selection=selection)
        result = tools.model_iterator(canonical='titanic', marker_col='marker', selection=selection)
        self.assertEqual(control.shape[0], result.shape[0])
        # with iteration
        result = tools.model_iterator(canonical='titanic', marker_col='marker', iter_stop=3)
        self.assertCountEqual([0,1,2], result['marker'].value_counts().index.to_list())
        # with actions
        actions = {2: (tools.action2dict(method='get_category', selection=[4,5]))}
        result = tools.model_iterator(canonical='titanic', marker_col='marker', iter_stop=3, iteration_actions=actions)
        self.assertCountEqual([0,1,4,5], result['marker'].value_counts().index.to_list())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
