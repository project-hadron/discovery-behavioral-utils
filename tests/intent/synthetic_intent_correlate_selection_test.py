import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentCorrelateSelectionTest(unittest.TestCase):

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

    def test_action_value(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1,4,2,1,6,1]
        selection = [tools.select2dict(column='letters', condition="== 'A'")]
        result = tools.correlate_selection(df, selection=selection, action=9)
        self.assertEqual([9.0, None, 9.0, None, None, None], result)
        # 2 selections
        selection = [tools.select2dict(column='letters', condition="== 'A'"),
                     tools.select2dict(column='letters', condition="'B'", operator='==', logic='OR')]
        result = tools.correlate_selection(df, selection=selection, action=9)
        self.assertEqual([9.0, 9.0, 9.0, 9.0, 9.0, None], result)
        # three selections
        selection = [tools.select2dict(column='letters', condition="== 'A'"),
                     tools.select2dict(column='letters', condition="'B'", operator='==', logic='OR'),
                     tools.select2dict(column='value', condition="==1", logic='AND')]
        result = tools.correlate_selection(df, selection=selection, action=9)
        self.assertEqual([9.0, None, None, 9.0, None, None], result)

    def test_action_header(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1,4,2,1,6,1]
        selection = [tools.select2dict(column='letters', condition="== 'A'")]
        action = tools.action2dict(method="@header", header='value')
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual([1, -1, 2, -1, -1, -1], result)

    def test_action_method(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1,4,2,1,6,1]
        selection = [tools.select2dict(column='value', condition=">1")]
        action = tools.action2dict(method="get_category", selection=['X'])
        default_action = tools.action2dict(method="get_category", selection=['M'])
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=default_action)
        self.assertEqual(['M', 'X', 'X', 'M', 'X', 'M'], result)

    def test_action_constant(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1,4,2,1,6,1]
        selection = [tools.select2dict(column='value', condition=">1")]
        action = tools.action2dict(method="@constant", value='14')
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual([-1, '14', '14', -1, '14', -1], result)

    def test_action_eval(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1,4,2,1,6,1]
        selection = [tools.select2dict(column='value', condition=">1")]
        action = tools.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual([-1, 8, 8, -1, 8, -1], result)

    def test_action_correlate(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='value', condition=">1")]
        action = tools.action2dict(method='correlate_numbers', header='value', offset=0.8, multiply_offset=True)
        default_action = tools.action2dict(method="@header", header='value')
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=default_action)
        self.assertEqual([1.0, 3.2, 1.6, 1.0, 4.8, 1.0], result)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
