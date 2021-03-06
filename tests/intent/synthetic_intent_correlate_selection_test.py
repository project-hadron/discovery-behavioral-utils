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

    def test_selection_function(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', None, 'B', None, 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='letters', condition="@.isna()")]
        result = tools.correlate_selection(df, selection=selection, action='N/A')
        self.assertEqual([None, None, 'N/A', None, 'N/A', None], result)

    def test_selection_complex(self):
        df = pd.DataFrame()
        df['s1'] = pd.Series(list('AAAABBBBCCCCDDDD'))
        df['s2'] = pd.Series(list('ABCDABCDABCDABCD'))
        df['s3'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
        # single column
        selection = [self.tools.select2dict(column='s3', condition="(@ != 2) & ([x not in [4,5,6,7] for x in @])")]
        action = self.tools.action2dict(method='@constant', value=1)
        default = self.tools.action2dict(method='@constant', value=0)
        df['l1'] = self.tools.correlate_selection(df, selection=selection, action=action, default_action=default)
        result = df[df['l1'] == 1].loc[:, 's3']
        self.assertEqual([1, 3, 8, 1, 3, 8], list(result.values))
        # multiple column
        selection = [self.tools.select2dict(column='s1', condition="@ != 'A'"),
                     self.tools.select2dict(column='s2', condition="@ == 'B'", logic='AND')]
        action = self.tools.action2dict(method='@constant', value=1)
        default = self.tools.action2dict(method='@constant', value=0)
        df['l2'] = self.tools.correlate_selection(df, selection=selection, action=action, default_action=default)
        result = df[df['l2'] == 1].loc[:, 's1']
        self.assertEqual(['B', 'C', 'D'], list(result.values))
        result = df[df['l2'] == 1].loc[:, 's2']
        self.assertEqual(['B', 'B', 'B'], list(result.values))

    def test_action_value(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='letters', condition="@ == 'A'")]
        result = tools.correlate_selection(df, selection=selection, action=9)
        self.assertEqual([9.0, None, 9.0, None, None, None], result)
        # 2 selections
        selection = [tools.select2dict(column='letters', condition="@ == 'A'"),
                     tools.select2dict(column='letters', condition="@ == 'B'", logic='OR')]
        result = tools.correlate_selection(df, selection=selection, action=9)
        self.assertEqual([9., 9., 9., 9., 9., None], result)
        # three selections
        selection = [tools.select2dict(column='letters', condition="@ == 'A'"),
                     tools.select2dict(column='letters', condition="@ == 'B'", logic='OR'),
                     tools.select2dict(column='value', condition="@ ==1", logic='AND')]
        result = tools.correlate_selection(df, selection=selection, action=9)
        self.assertEqual([9.0, None, None, 9.0, None, None], result)

    def test_action_header(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='letters', condition="@ == 'A'")]
        action = tools.action2dict(method="@header", header='value')
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual([1, -1, 2, -1, -1, -1], result)

    def test_action_sample(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        selection = [tools.select2dict(column='letters', condition="@ == 'A'")]
        action = tools.action2dict(method="@sample", name='us_states', shuffle=False)
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual(['AA', -1, 'AE', -1, -1, -1], result)

    def test_action_method(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='value', condition="@ >1")]
        action = tools.action2dict(method="get_category", selection=['X'])
        default_action = tools.action2dict(method="get_category", selection=['M'])
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=default_action)
        self.assertEqual(['M', 'X', 'X', 'M', 'X', 'M'], result)

    def test_action_constant(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='value', condition="@ > 1")]
        action = tools.action2dict(method="@constant", value='14')
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual([-1, '14', '14', -1, '14', -1], result)

    def test_action_eval(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='value', condition="@ > 1")]
        action = tools.action2dict(method="@eval", code_str='sum(values)', values=[1, 4, 2, 1])
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=-1)
        self.assertEqual([-1, 8, 8, -1, 8, -1], result)

    def test_action_correlate(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1, 4, 2, 1, 6, 1]
        selection = [tools.select2dict(column='value', condition="@ > 1")]
        action = tools.action2dict(method='correlate_numbers', header='value', offset=0.8, multiply_offset=True)
        default_action = tools.action2dict(method="@header", header='value')
        result = tools.correlate_selection(df, selection=selection, action=action, default_action=default_action)
        self.assertEqual([1.0, 3.2, 1.6, 1.0, 4.8, 1.0], result)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))

    def test_selection_logic_two_elements(self):
        df = pd.DataFrame()
        df['s1'] = pd.Series(list('AAAABBBBCCCCDDDD'))
        df['s2'] = pd.Series(list('ABCDABCDABCDABCD'))
        df['s3'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
        # A AND B
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'")
        c2 = self.tools.select2dict(column='s2', condition="@ == 'B'", logic='AND')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c2])]
        self.assertEqual([1], result.index.to_list())
        # A NAND B
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'")
        c2 = self.tools.select2dict(column='s2', condition="@ == 'B'", logic='NAND')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c2])]
        self.assertEqual([0] + list(range(2, 16)), result.index.to_list())
        # A OR [C, D]
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'")
        c4 = self.tools.select2dict(column='s2', condition="@.isin(['C', 'D'])", logic='OR')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c4])]
        self.assertEqual([0,1,2,3,6,7,10,11,14,15], result.index.to_list())
        # A NOR [C, D]
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'")
        c4 = self.tools.select2dict(column='s2', condition="@.isin(['C', 'D'])", logic='NOR')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c4])]
        self.assertEqual([4, 5, 8, 9, 12, 13], result.index.to_list())
        # A NOT B
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'", logic='AND')
        c2 = self.tools.select2dict(column='s2', condition="@ == 'B'", logic='NOT')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c2])]
        self.assertEqual([0,2,3], result.index.to_list())
        # A XOR B
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'")
        c2 = self.tools.select2dict(column='s2', condition="@ == 'B'", logic='XOR')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c2])]
        self.assertEqual([0, 2, 3, 5, 9, 13], result.index.to_list())
        # !(A AND B)
        c1 = self.tools.select2dict(column='s3', condition="@ < 8")
        c2 = self.tools.select2dict(column='s3', condition="@ > 5", logic='AND')
        result = df.iloc[self.tools._selection_index(df, selection=[[c1, c2], 'NOT'])]
        self.assertEqual([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 15], result.index.to_list())

    def test_selection_logic_three_elements(self):
        df = pd.DataFrame()
        df['s1'] = pd.Series(list('AAAABBBBCCCCDDDD'))
        df['s2'] = pd.Series(list('ABCDABCDABCDABCD'))
        df['s3'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'")
        c2 = self.tools.select2dict(column='s2', condition="@ == 'B'", logic='OR')
        c3 = self.tools.select2dict(column='s3', condition="@ < 4", logic='AND')
        result = df.iloc[self.tools._selection_index(df, selection=[c1, c2, c3])]
        self.assertEqual([0, 1, 2, 9], result.index.to_list())

    def test_selection_logic_nested(self):
        df = pd.DataFrame()
        df['s1'] = pd.Series(list('AAAABBBBCCCCDDDD'))
        df['s2'] = pd.Series(list('ABCDABCDABCDABCD'))
        df['s3'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
        c1 = self.tools.select2dict(column='s1', condition="@ == 'A'", logic='AND')
        c2 = self.tools.select2dict(column='s2', condition="@ == 'B'", logic='OR')
        c3 = self.tools.select2dict(column='s3', condition="@ < 4", logic='AND')
        result = df.iloc[self.tools._selection_index(df, selection=[[c1, c2], c3])]
        self.assertEqual([0, 1, 2, 9], result.index.to_list())
        result = df.iloc[self.tools._selection_index(df, selection=[c3, [c1, c2]])]
        self.assertEqual([0, 1, 2, 9], result.index.to_list())
        A = self.tools.select2dict(column='s3', condition="(@ > 2)", logic='AND')
        B = self.tools.select2dict(column='s3', condition="(@ < 5)", logic='AND')
        C = self.tools.select2dict(column='s3', condition="@ == 8", logic='OR')
        selection = [[A, B], C]
        result = df.iloc[self.tools._selection_index(df, selection=selection)]
        self.assertEqual([2, 3, 7, 10, 11, 15], result.index.to_list())


if __name__ == '__main__':
    unittest.main()
