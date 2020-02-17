from pprint import pprint
import pandas as pd
import unittest
from ds_foundation.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.transition.discovery import DataAnalytics, DataDiscovery as Discovery
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel


class ControlPropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str):
        # set additional keys
        root_keys = []
        knowledge_keys = []
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys)


class AnalysisTest(unittest.TestCase):

    def setUp(self):
        self.pm = ControlPropertyManager('test_abstract_properties')
        self.tools = SyntheticIntentModel(property_manager=self.pm, default_save_intent=False)
        self.pm.reset_all()

    def tearDown(self):
        pass

    def test_associate_analysis_from_discovery(self):
        """Basic smoke test"""
        df = pd.DataFrame()
        df['cat'] = self.tools.get_category(selection=list('ABC'), quantity=0.9, size=100)
        df['values'] = self.tools.get_number(from_value=20, dominant_values=0, dominant_percent=0.6, size=100)
        # discover
        associate = [{'cat': {'dtype': 'category'}, 'values': {'granularity': 5, 'precision': 3, 'lower': 0.001}}]
        analysis = Discovery.analyse_association(df, columns_list=associate)
        # build
        sample_size=173
        result = self.tools.associate_analysis(analysis, size=sample_size)
        self.assertCountEqual(['cat', 'values'], result.keys())
        for key in result.keys():
            self.assertEqual(sample_size, len(result.get(key)))
