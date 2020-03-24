import unittest
import os
import shutil

from ds_behavioral import SyntheticBuilder
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentCorrelateTest(unittest.TestCase):

    def setUp(self):
        os.environ['AISTAC_PM_PATH'] = os.path.join('work', 'config')
        os.environ['AISTAC_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['AISTAC_PM_PATH'])
            os.makedirs(os.environ['AISTAC_DEFAULT_PATH'])
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

    def test_runs(self):
        """Basic smoke test"""
        im = SyntheticBuilder.from_env('tester', default_save=False, default_save_intent=False,
                                       reset_templates=False).intent_model
        self.assertTrue(SyntheticIntentModel, type(im))

    def test_correlate_number(self):
        tools = self.tools
        numbers = [1,2,3,4,5,6,7,8,9,0]
        result = tools.correlate_numbers(numbers, label='corr_numbers', precision=0)
        self.assertEqual(numbers, result)
        result = tools.correlate_numbers(numbers, offset=1, label='corr_numbers', precision=0)
        self.assertEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 1], result)
        result = tools.correlate_numbers(numbers, spread=2, label='corr_numbers', precision=1)
        for index in range(len(result)):
            loss = abs(numbers[index] - result[index])
            self.assertLessEqual(loss, 2)
        print(result)

    def test_correlate_categories(self):
        tools = self.tools
        categories = list("ABCDE")
        correlation = ['A', 'B']
        action = {0: {'action': 'F'}, 1: {'action': 'G'}}
        result = tools.correlate_categories(categories, correlations=correlation, actions=action, value_type='category', label='letters')
        print(result)


if __name__ == '__main__':
    unittest.main()
