import matplotlib

matplotlib.use("TkAgg")

import unittest
import warnings

from ds_behavioral.sample.sample_data import MappedSample

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
        result = MappedSample.companies_fortune1000(size=10)
        print(result)

if __name__ == '__main__':
    unittest.main()
