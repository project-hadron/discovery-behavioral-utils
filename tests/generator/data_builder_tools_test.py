import datetime
import unittest
from ds_behavioral import DataBuilderTools as tools
from matplotlib import dates as mdates


class MyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dates(self):
        result = tools.get_datetime('01/01/2018', '01/01/2019', as_num=True, size=1000)
        self.assertEqual(1000, len(result))
        # get the values at the edge
        value_min = tools.get_datetime('01/01/2018', '01/01/2018', as_num=True, size=1)[0]
        value_max = tools.get_datetime('01/01/2019', '01/01/2019', as_num=True, size=1)[0]
        self.assertLessEqual(value_min, min(result))
        self.assertGreaterEqual(value_max, max(result))
        result = tools.get_datetime('01/01/2018', '01/01/2018')
        self.assertEqual([datetime.datetime(2018, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)], result)
        # test date_format
        result = tools.get_datetime('01/01/2018', '01/01/2018', date_format="%Y-%m-%d")
        self.assertEqual(['2018-01-01'], result)
        # test ignore time
        result = tools.get_datetime('01/01/2018T01:01:01', '01/01/2018T23:59:59', ignore_time=True)
        self.assertEqual([datetime.datetime(2018, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)], result)
        # test at_most
        result = tools.get_datetime('01/01/2018', '01/04/2018', ignore_time=True, at_most=1, size=3, date_format="%Y-%m-%d")
        self.assertEqual(3, len(result))
        self.assertCountEqual(['2018-01-01', '2018-01-03', '2018-01-02'], result)


if __name__ == '__main__':
    unittest.main()
