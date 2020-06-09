import unittest
import os
import shutil
import pandas as pd
from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentTest(unittest.TestCase):

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

    def test_canonical_smoke(self):
        result = self.tools.associate_canonical(pd.DataFrame(), associations=[], actions={})
        self.assertEqual([], result)

    def test_canonical(self):
        tools = self.tools
        df = pd.DataFrame()
        df['letters'] = ['A', 'B', 'A', 'B', 'B', 'C']
        df['value'] = [1,4,2,1,4,1]
        df['date'] = ['2019/01/04', '2019/01/06', '2019/01/02', '2019/01/04', '2019/01/06', '2019/01/03']
        associations = [{'value': {'expect': 'number', 'value': [2, 7]}}]
        actions = {0: {'action': 9}}
        result = tools.associate_canonical(df, associations=associations, actions=actions)
        self.assertEqual([None, 9, 9, None, 9, None], result)
        associations = []
        actions = {}
        result = tools.associate_canonical(df, associations=associations, actions=actions, default_header='letters')
        print(result)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
