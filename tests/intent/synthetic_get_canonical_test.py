import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticGetCanonicalTest(unittest.TestCase):

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

        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'contracts')
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

    def test_dataframe(self):
        tools = SyntheticBuilder.from_memory().tools
        df = pd.DataFrame(data={'A': list('12345')})
        result = tools._get_canonical(data=df)
        self.assertDictEqual(df.to_dict(), result.to_dict())

    def test_str(self):
        builder = SyntheticBuilder.from_memory()
        tools = builder.tools
        df = pd.DataFrame(data={'A': list('12345')})
        builder.add_connector_persist(connector_name='test', uri_file='test.pickle')
        builder.save_canonical(connector_name='test', canonical=df)
        result = tools._get_canonical(data='test')
        self.assertDictEqual(df.to_dict(), result.to_dict())

    def test_list(self):
        builder = SyntheticBuilder.from_memory()
        tools = builder.tools
        sample = list('12345')
        result = tools._get_canonical(data=sample)
        self.assertEqual(sample, result['default'].to_list())
        result = tools._get_canonical(data=sample, header='sample')
        self.assertEqual(sample, result['sample'].to_list())
        sample = pd.Series(sample)
        result = tools._get_canonical(data=sample, header='sample')
        self.assertEqual(sample.to_list(), result['sample'].to_list())

    def test_dict_generate(self):
        builder = SyntheticBuilder.from_env('generator', has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['gender'] = tools.get_category(selection=['M', 'F'], column_name='gender')
        df['age'] = tools.get_number(from_value=18, to_value=90, column_name='age')
        target = {'method': '@generate', 'task_name': 'generator'}
        result = tools._get_canonical(data=target)
        self.assertCountEqual(['age', 'gender'], result.columns.to_list())
        target = {'method': '@generate', 'task_name': 'generator', 'size': 100}
        result = tools._get_canonical(data=target)
        self.assertCountEqual(['age', 'gender'], result.columns.to_list())
        self.assertEqual(100, result.shape[0])
        selection = [tools.select2dict(column='gender', condition="@=='M'")]
        target = {'method': '@generate', 'task_name': 'generator', 'size': 100, 'selection': selection}
        result = tools._get_canonical(data=target)
        self.assertGreater(result.shape[0], 0)
        self.assertEqual(0, (result[result['gender'] == 'F']).shape[0])

    def test_dict_generate_remote(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        canonical = tools.canonical2dict(method='@empty', size=1000)
        other = tools.canonical2dict(method='@generate', task_name='members', uri_pm_repo='https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/healthcare/factory/members/')
        result = builder.intent_model.model_concat(canonical=canonical, other=other, as_rows=False,
                                                   headers=['member_id', 'state', 'prev_flu_shot', 'age', 'channel_pref'],
                                                   column_name='member_reference')
        print(result.columns)


    def test_dict_method(self):
        builder = SyntheticBuilder.from_env('generator', has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        action = tools.canonical2dict(method='model_sample_map', canonical=tools.action2dict(method='@empty', size=100),
                                      sample_map='us_persona', female_bias=0.3)
        result = tools._get_canonical(data=action)
        self.assertEqual((100, 5), result.shape)
        self.assertEqual(30, result['gender'].value_counts().loc['F'])

    def test_dict_empty(self):
        builder = SyntheticBuilder.from_env('generator', has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        action = tools.canonical2dict(method='@empty')
        result = tools._get_canonical(data=action)
        self.assertEqual((0, 0), result.shape)
        action = tools.canonical2dict(method='@empty', size=100)
        result = tools._get_canonical(data=action)
        self.assertEqual((100, 0), result.shape)
        action = tools.canonical2dict(method='@empty', size=100)
        result = tools._get_canonical(data=action)
        self.assertEqual((100, 0), result.shape)
        action = tools.canonical2dict(method='@empty', size=100, headers=['A', 'B', 'C'])
        result = tools._get_canonical(data=action)
        self.assertEqual((100, 3), result.shape)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
