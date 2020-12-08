import unittest

from tests.components.synthetic_builder_test import SyntheticBuilderTest
from tests.intent.synthetic_intent_analysis_test import SyntheticIntentAnalysisTest
from tests.intent.synthetic_intent_correlate_selection_test import SyntheticIntentCorrelateSelectionTest
from tests.intent.synthetic_intent_correlate_test import SyntheticIntentCorrelateTest
from tests.intent.synthetic_intent_get_test import SyntheticIntentGetTest
from tests.intent.synthetic_pipeline_test import SyntheticPipelineTest
from tests.intent.synthetic_weighting_test import SyntheticWeightingTest
from tests.intent.synthetic_get_canonical_test import SyntheticGetCanonicalTest


# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentAnalysisTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentCorrelateSelectionTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentCorrelateTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentGetTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticPipelineTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticWeightingTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticGetCanonicalTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticBuilderTest))


# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
