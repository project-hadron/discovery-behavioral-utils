import unittest
import pandas as pd
from ds_behavioral.sample.sample_data import Sample, MappedSample


class MyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_select_list(self):
        original = list(range(50))
        # no params
        result = Sample._select_list(selection=original.copy())
        self.assertEqual(len(original), len(result))
        self.assertNotEqual(original, result)
        # no shuffle
        result = Sample._select_list(selection=original.copy(), shuffle=False)
        self.assertEqual(len(original), len(result))
        self.assertEqual(original, result)
        # size
        size = 5
        result = Sample._select_list(selection=original.copy(), shuffle=False, size=size)
        self.assertEqual(size, len(result))
        self.assertEqual(original[:5], result)
        # no_seed
        no_seed = Sample._select_list(selection=original.copy())
        result = Sample._select_list(selection=original.copy())
        self.assertNotEqual(no_seed, result)
        # seed
        seed = Sample._select_list(selection=original.copy(), seed=31)
        result = Sample._select_list(selection=original.copy(), seed=31)
        self.assertEqual(seed, result)


    def test_get_constant(self):
        # map
        result = Sample._get_constant(reference='map_us_surname_rank')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(2, result.shape[1])
        self.assertTrue(result.shape[0] > 0)
        # lookup
        result = Sample._get_constant(reference='lookup_us_street_suffix')
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_maps(self):
        size = 200000
        for name in MappedSample().__dir__():
            result = eval(f"MappedSample.{name}(size={size})")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(size, result.shape[0])

    def test_lookups(self):
        size = 200000
        for name in Sample().__dir__():
            result = eval(f"Sample.{name}(size={size})")
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), size)


if __name__ == '__main__':
    unittest.main()
