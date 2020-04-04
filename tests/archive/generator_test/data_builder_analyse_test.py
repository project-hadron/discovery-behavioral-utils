from pprint import pprint

import matplotlib
import pandas as pd

matplotlib.use("TkAgg")

import unittest
import warnings

from ds_behavioral import DataBuilderTools
from ds_discovery.transition.discovery import DataDiscovery as discover


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class AnalysisTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_runs(self):
        """Basic smoke test"""
        DataBuilderTools()

    def test_anaysis(self):
        tools = DataBuilderTools
        df = pd.DataFrame()
        df['cat'] = tools.get_category(selection=list('ABC'), quantity=0.9, size=1000)
        df['values'] = tools.get_number(from_value=20, dominant_values=0, dominant_percent=0.6, size=1000)
        associate = [{'cat': {'dtype': 'category'}},{'values': {'granularity': 5, 'precision': 3, 'lower': 0.001}}]
        analysis = discover.analyse_association(df, columns_list=associate)
        sample_size=1973
        result = tools.model_analysis(analysis, size=sample_size)
        self.assertCountEqual(['cat', 'values'], result.keys())
        for key in result.keys():
            self.assertEqual(sample_size, len(result.get(key)))

