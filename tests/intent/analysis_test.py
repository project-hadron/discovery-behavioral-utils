from pprint import pprint
import pandas as pd
import unittest
from aistac.properties.abstract_properties import AbstractPropertyManager
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
        associate = [{'cat': {'dtype': 'category'}, 'values': {'dtype': 'category','granularity': 5, 'precision': 3}}]
        analysis = Discovery.analyse_association(df, columns_list=associate)
        # build
        sample_size=173
        result = self.tools.associate_analysis(analysis, size=sample_size)
        self.assertCountEqual(['cat', 'values'], result.keys())
        for key in result.keys():
            self.assertEqual(sample_size, len(result.get(key)))

    def test_associate_from_dict(self):
        analysis = {'income': {'associate': 'income',
                               'analysis': {'intent': {'selection': [(0, 1, 'both')],
                                                       'granularity': 1,
                                                       'dtype': 'number',
                                                       'precision': 0,
                                                       'lower': 0,
                                                       'upper': 1,
                                                       'weighting_precision': 2},
                                            'patterns': {'weight_pattern': [100.0],
                                                         'weight_mean': [0.239],
                                                         'weight_std': [0.182],
                                                         'sample_distribution': [48842],
                                                         'dominant_values': [0],
                                                         'dominance_weighting': [100.0],
                                                         'dominant_percent': 5},
                                            'stats': {'nulls_percent': 0.0,
                                                      'sample': 48842,
                                                      'outlier_percent': 0.0,
                                                      'mean': 0.24,
                                                      'var': 0.18,
                                                      'skew': 1.22,
                                                      'kurtosis': -0.51}}}}
        sample_size=20
        result = self.tools.associate_analysis(analysis, size=sample_size)
        print(pd.DataFrame(result))

        

