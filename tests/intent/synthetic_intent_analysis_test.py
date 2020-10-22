from pprint import pprint
import pandas as pd
import unittest
from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.components.discovery import DataAnalytics, DataDiscovery as Discovery
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel


class ControlPropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str):
        # set additional keys
        root_keys = []
        knowledge_keys = []
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, username='default')


class SyntheticIntentAnalysisTest(unittest.TestCase):

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
        df['values'] = self.tools.get_number(range_value=20, dominant_values=0, dominant_percent=0.6, size=100)
        # discover
        associate = [{'cat': {'dtype': 'category'}, 'values': {'dtype': 'category','granularity': 5, 'precision': 3}}]
        analysis = Discovery.analyse_association(df, columns_list=associate)
        # build
        sample_size=173
        result = self.tools.model_analysis(analysis, size=sample_size)
        self.assertCountEqual(['cat', 'values'], result.keys())
        for key in result.keys():
            self.assertEqual(sample_size, len(result.get(key)))

    def test_associate_from_dict(self):
        analysis = {'age': {'associate': 'age',
                            'analysis': {'intent': {'selection': [(17, 90, 'both')],
                                                    'granularity': 1,
                                                    'dtype': 'number',
                                                    'precision': 0,
                                                    'lower': 17,
                                                    'upper': 90,
                                                    'weighting_precision': 2},
                                         'patterns': {'weight_pattern': [100.0],
                                                      'weight_mean': [38.644],
                                                      'weight_std': [187.978],
                                                      'sample_distribution': [48842],
                                                      'dominant_values': [36],
                                                      'dominance_weighting': [100.0],
                                                      'dominant_percent': 2.76},
                                         'stats': {'nulls_percent': 0.0,
                                                   'sample': 48842,
                                                   'outlier_percent': 0.0,
                                                   'mean': 38.64,
                                                   'var': 187.98,
                                                   'skew': 0.56,
                                                   'kurtosis': -0.18}}},
                    'workclass': {'associate': 'workclass',
                                  'analysis': {'intent': {'selection': ['workclass_Private',
                                                                        'workclass_Self-emp-not-inc',
                                                                        'workclass_Local-gov',
                                                                        'workclass_unknown',
                                                                        'workclass_State-gov',
                                                                        'workclass_Self-emp-inc',
                                                                        'workclass_Federal-gov',
                                                                        'workclass_Without-pay',
                                                                        'workclass_Never-worked'],
                                                          'dtype': 'category',
                                                          'upper': 69.42,
                                                          'lower': 0.02,
                                                          'granularity': 9,
                                                          'weighting_precision': 2},
                                               'patterns': {'weight_pattern': [69.42,
                                                                               7.91,
                                                                               6.42,
                                                                               5.73,
                                                                               4.06,
                                                                               3.47,
                                                                               2.93,
                                                                               0.04,
                                                                               0.02],
                                                            'sample_distribution': [33906,
                                                                                    3862,
                                                                                    3136,
                                                                                    2799,
                                                                                    1981,
                                                                                    1695,
                                                                                    1432,
                                                                                    21,
                                                                                    10]},
                                               'stats': {'nulls_percent': 0.0, 'outlier_percent': 0.0,
                                                         'sample': 48842}}},
                    'fnlwgt': {'associate': 'fnlwgt',
                               'analysis': {'intent': {'selection': [(12285, 1490400, 'both')],
                                                       'granularity': 1,
                                                       'dtype': 'number',
                                                       'precision': 0,
                                                       'lower': 12285,
                                                       'upper': 1490400,
                                                       'weighting_precision': 2},
                                            'patterns': {'weight_pattern': [100.0],
                                                         'weight_mean': [189664.135],
                                                         'weight_std': [11152210185.575],
                                                         'sample_distribution': [48842],
                                                         'dominant_values': [203488],
                                                         'dominance_weighting': [100.0],
                                                         'dominant_percent': 0.04},
                                            'stats': {'nulls_percent': 0.0,
                                                      'sample': 48842,
                                                      'outlier_percent': 0.0,
                                                      'mean': 189664.13,
                                                      'var': 11152210185.57,
                                                      'skew': 1.44,
                                                      'kurtosis': 6.06}}},
                    'marital-status': {'associate': 'marital-status',
                                       'analysis': {'intent': {'selection': ['marital-status_Married-civ-spouse',
                                                                             'marital-status_Never-married',
                                                                             'marital-status_Divorced',
                                                                             'marital-status_Separated',
                                                                             'marital-status_Widowed',
                                                                             'marital-status_Married-spouse-absent',
                                                                             'marital-status_Married-AF-spouse'],
                                                               'dtype': 'category',
                                                               'upper': 45.82,
                                                               'lower': 0.08,
                                                               'granularity': 7,
                                                               'weighting_precision': 2},
                                                    'patterns': {
                                                        'weight_pattern': [45.82, 33.0, 13.58, 3.13, 3.11, 1.29, 0.08],
                                                        'sample_distribution': [22379, 16117, 6633, 1530, 1518, 628,
                                                                                37]},
                                                    'stats': {'nulls_percent': 0.0, 'outlier_percent': 0.0,
                                                              'sample': 48842}}},
                    'occupation': {'associate': 'occupation',
                                   'analysis': {'intent': {'selection': ['occupation_Prof-specialty',
                                                                         'occupation_Craft-repair',
                                                                         'occupation_Exec-managerial',
                                                                         'occupation_Adm-clerical',
                                                                         'occupation_Sales',
                                                                         'occupation_Other-service',
                                                                         'occupation_Machine-op-inspct',
                                                                         'occupation_unknown',
                                                                         'occupation_Transport-moving',
                                                                         'occupation_Handlers-cleaners',
                                                                         'occupation_Farming-fishing',
                                                                         'occupation_Tech-support',
                                                                         'occupation_Protective-serv',
                                                                         'occupation_Priv-house-serv',
                                                                         'occupation_Armed-Forces'],
                                                           'dtype': 'category',
                                                           'upper': 12.64,
                                                           'lower': 0.03,
                                                           'granularity': 15,
                                                           'weighting_precision': 2},
                                                'patterns': {'weight_pattern': [12.64,
                                                                                12.51,
                                                                                12.46,
                                                                                11.49,
                                                                                11.27,
                                                                                10.08,
                                                                                6.19,
                                                                                5.75,
                                                                                4.82,
                                                                                4.24,
                                                                                3.05,
                                                                                2.96,
                                                                                2.01,
                                                                                0.5,
                                                                                0.03],
                                                             'sample_distribution': [6172,
                                                                                     6112,
                                                                                     6086,
                                                                                     5611,
                                                                                     5504,
                                                                                     4923,
                                                                                     3022,
                                                                                     2809,
                                                                                     2355,
                                                                                     2072,
                                                                                     1490,
                                                                                     1446,
                                                                                     983,
                                                                                     242,
                                                                                     15]},
                                                'stats': {'nulls_percent': 0.0, 'outlier_percent': 0.0,
                                                          'sample': 48842}}},
                    'relationship': {'associate': 'relationship',
                                     'analysis': {'intent': {'selection': ['relationship_Husband',
                                                                           'relationship_Not-in-family',
                                                                           'relationship_Own-child',
                                                                           'relationship_Unmarried',
                                                                           'relationship_Wife',
                                                                           'relationship_Other-relative'],
                                                             'dtype': 'category',
                                                             'upper': 40.37,
                                                             'lower': 3.08,
                                                             'granularity': 6,
                                                             'weighting_precision': 2},
                                                  'patterns': {
                                                      'weight_pattern': [40.37, 25.76, 15.52, 10.49, 4.77, 3.08],
                                                      'sample_distribution': [19716, 12583, 7581, 5125, 2331, 1506]},
                                                  'stats': {'nulls_percent': 0.0, 'outlier_percent': 0.0,
                                                            'sample': 48842}}},
                    'race': {'associate': 'race',
                             'analysis': {'intent': {'selection': ['race_White',
                                                                   'race_Black',
                                                                   'race_Asian-Pac-Islander',
                                                                   'race_Amer-Indian-Eskimo',
                                                                   'race_Other'],
                                                     'dtype': 'category',
                                                     'upper': 85.5,
                                                     'lower': 0.83,
                                                     'granularity': 5,
                                                     'weighting_precision': 2},
                                          'patterns': {'weight_pattern': [85.5, 9.59, 3.11, 0.96, 0.83],
                                                       'sample_distribution': [41762, 4685, 1519, 470, 406]},
                                          'stats': {'nulls_percent': 0.0, 'outlier_percent': 0.0, 'sample': 48842}}},
                    'gender': {'associate': 'gender',
                               'analysis': {'intent': {'selection': ['gender_Male', 'gender_Female'],
                                                       'dtype': 'category',
                                                       'upper': 66.85,
                                                       'lower': 33.15,
                                                       'granularity': 2,
                                                       'weighting_precision': 2},
                                            'patterns': {'weight_pattern': [66.85, 33.15],
                                                         'sample_distribution': [32650, 16192]},
                                            'stats': {'nulls_percent': 0.0, 'outlier_percent': 0.0, 'sample': 48842}}},
                    'capital-gain': {'associate': 'capital-gain',
                                     'analysis': {'intent': {'selection': [(0, 99999, 'both')],
                                                             'granularity': 1,
                                                             'dtype': 'number',
                                                             'precision': 0,
                                                             'lower': 0,
                                                             'upper': 99999,
                                                             'weighting_precision': 2},
                                                  'patterns': {'weight_pattern': [100.0],
                                                               'weight_mean': [1079.068],
                                                               'weight_std': [55532588.036],
                                                               'sample_distribution': [48842],
                                                               'dominant_values': [0],
                                                               'dominance_weighting': [100.0],
                                                               'dominant_percent': 91.74},
                                                  'stats': {'nulls_percent': 0.0,
                                                            'sample': 48842,
                                                            'outlier_percent': 0.0,
                                                            'mean': 1079.07,
                                                            'var': 55532588.04,
                                                            'skew': 11.89,
                                                            'kurtosis': 152.69}}},
                    'capital-loss': {'associate': 'capital-loss',
                                     'analysis': {'intent': {'selection': [(0, 4356, 'both')],
                                                             'granularity': 1,
                                                             'dtype': 'number',
                                                             'precision': 0,
                                                             'lower': 0,
                                                             'upper': 4356,
                                                             'weighting_precision': 2},
                                                  'patterns': {'weight_pattern': [100.0],
                                                               'weight_mean': [87.502],
                                                               'weight_std': [162412.669],
                                                               'sample_distribution': [48842],
                                                               'dominant_values': [0],
                                                               'dominance_weighting': [100.0],
                                                               'dominant_percent': 95.33},
                                                  'stats': {'nulls_percent': 0.0,
                                                            'sample': 48842,
                                                            'outlier_percent': 0.0,
                                                            'mean': 87.5,
                                                            'var': 162412.67,
                                                            'skew': 4.57,
                                                            'kurtosis': 20.01}}},
                    'hours-per-week': {'associate': 'hours-per-week',
                                       'analysis': {'intent': {'selection': [(1, 99, 'both')],
                                                               'granularity': 1,
                                                               'dtype': 'number',
                                                               'precision': 0,
                                                               'lower': 1,
                                                               'upper': 99,
                                                               'weighting_precision': 2},
                                                    'patterns': {'weight_pattern': [100.0],
                                                                 'weight_mean': [40.422],
                                                                 'weight_std': [153.548],
                                                                 'sample_distribution': [48842],
                                                                 'dominant_values': [40],
                                                                 'dominance_weighting': [100.0],
                                                                 'dominant_percent': 46.69},
                                                    'stats': {'nulls_percent': 0.0,
                                                              'sample': 48842,
                                                              'outlier_percent': 0.0,
                                                              'mean': 40.42,
                                                              'var': 153.55,
                                                              'skew': 0.24,
                                                              'kurtosis': 2.95}}},
                    'income': {'associate': 'income',
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
                                                         'dominant_percent': 76.07},
                                            'stats': {'nulls_percent': 0.0,
                                                      'sample': 48842,
                                                      'outlier_percent': 0.0,
                                                      'mean': 0.24,
                                                      'var': 0.18,
                                                      'skew': 1.22,
                                                      'kurtosis': -0.51}}}}
        sample_size = 10000
        result = self.tools.model_analysis(analysis, size=sample_size, save_intent=False)
        self.assertCountEqual(analysis.keys(), result.keys())
        for key, value in result.items():
            self.assertEqual(sample_size, len(value))

        

