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


class MyTestCase(unittest.TestCase):

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
        df['values'] = tools.get_number(from_value=20, dominant_value=0, dominance=0.6, size=1000)
        assosiate = [{'values': {'granularity': 0, 'precision': 3, 'lower': 0.001}}]
        analysis = discover.analyse_association(df, columns_list=assosiate)
        pprint(analysis)
        result = tools.associate_analysis(analysis, size=10)
        print(result)