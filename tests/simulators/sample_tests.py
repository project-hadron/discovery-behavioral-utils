import matplotlib
import pandas as pd

from ds_behavioral.intent.data_builder_tools import DataBuilderTools

matplotlib.use("TkAgg")

import unittest
import warnings


def ignore_warnings(message: str = None):
    def real_ignore_warnings(func):
        def do_test(self, *args, **kwargs):
            with warnings.catch_warnings():
                if isinstance(message, str) and len(message) > 0:
                    warnings.filterwarnings("ignore", message=message)
                else:
                    warnings.simplefilter("ignore")
                func(self, *args, **kwargs)

        return do_test

    return real_ignore_warnings


class MyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @ignore_warnings
    def test_runs(self):
        """Basic smoke test"""
        pass

    def test_map(self):
        tools = DataBuilderTools
        selection = ['M', 'F', 'U']
        weight_pattern = [5, 4, 1]
        select_index = tools.get_number(len(selection) - 1, weight_pattern=weight_pattern, size=1000)
        gender = [selection[i] for i in select_index]
        print(pd.Series(gender).value_counts())


if __name__ == '__main__':
    unittest.main()
