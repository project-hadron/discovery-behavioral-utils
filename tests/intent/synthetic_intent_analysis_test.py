import os
import shutil
from pprint import pprint
import pandas as pd
import numpy as np
import unittest

from aistac.properties.property_manager import PropertyManager
from ds_discovery import Transition
from ds_discovery.components.discovery import DataDiscovery as Discovery, DataDiscovery

from ds_behavioral import SyntheticBuilder
from ds_behavioral.components.commons import SyntheticCommons
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel


class SyntheticIntentAnalysisTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()
        self.builder: SyntheticBuilder = SyntheticBuilder.from_env('sample', has_contract=False)
        self.builder.setup_bootstrap()
        self.tools: SyntheticIntentModel = self.builder.tools

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_run_pipeline_with_analytics(self):
        builder: SyntheticBuilder = SyntheticBuilder.from_env('sample', has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        # load the sample dataset to analyse and rename columns
        clinical_health = 'https://assets.datacamp.com/production/repositories/628/datasets/444cdbf175d5fbf564b564bd36ac21740627a834/diabetes.csv'
        builder.add_connector_uri('clinical_health', uri=clinical_health)
        df_clinical = builder.load_canonical('clinical_health')

    def test_associate_analysis_from_discovery(self):
        df = pd.DataFrame()
        df['cat'] = self.tools.get_category(selection=list('ABC'), quantity=0.9, size=100, save_intent=False)
        df['values'] = self.tools.get_number(from_value=20, size=100, save_intent=False)
        # discover
        associate = [{'cat': {'dtype': 'category'}, 'values': {'dtype': 'category','granularity': 5, 'precision': 3}}]
        analysis = Discovery.analyse_association(df, columns_list=associate)
        # build
        empty_frame = self.tools.canonical2dict(method='@empty', size=1973)
        result = self.tools.model_analysis(canonical=empty_frame, analytics_model=analysis, save_intent=False)
        self.assertCountEqual(['cat', 'values'], result.keys())
        for key in result.keys():
            self.assertEqual(sample_size, len(result.get(key)))

    def test_associate_analysis_dominance(self):
        sample = pd.DataFrame()
        sample['values'] = [0,1,0,0,7,0,0,4,2,0,0,5,8,7,0,0]
        discover: DataDiscovery = Transition.from_memory().discover
        columns_list = [discover.analysis2dict(header='values', dtype='int', precision=0, exclude_dominant=True)]
        analysis_blob = discover.analyse_association(sample, columns_list=columns_list)
        builder = SyntheticBuilder.from_memory()
        canonical = builder.tools.canonical2dict(method='@empty', size=1000)
        df = builder.tools.model_analysis(canonical, analytics_model=analysis_blob, apply_bias=True)
        self.assertAlmostEqual(df['values'].value_counts().iloc[0]/df.shape[0], sample['values'].value_counts().iloc[0]/sample.shape[0], places=2)

    def test_associate_analysis_complex(self):
        builder = SyntheticBuilder.from_memory()
        clinical_health = 'https://assets.datacamp.com/production/repositories/628/datasets/444cdbf175d5fbf564b564bd36ac21740627a834/diabetes.csv'
        builder.add_connector_uri('clinical_health', uri=clinical_health)
        discover: DataDiscovery = Transition.from_memory().discover
        A = discover.analysis2dict(header='age', dtype='int', granularity=10.0, lower=21, upper=90)
        B = discover.analysis2dict(header='pregnancies')
        columns_list = [A, B]
        df_clinical = builder.load_canonical('clinical_health')
        analysis_blob = discover.analyse_association(df_clinical, columns_list=columns_list)
        canonical = builder.tools.canonical2dict(method='@empty', size=1973)
        df = builder.tools.model_analysis(canonical, analytics_model=analysis_blob, column_name='clinical')
        self.assertEqual((1973, 2), df.shape)
        pregnancies = SyntheticCommons.list_standardize(SyntheticCommons.list_formatter(df_clinical.pregnancies))
        low, high = discover.bootstrap_confidence_interval(pd.Series(pregnancies), func=np.mean)
        pregnancies = SyntheticCommons.list_standardize(SyntheticCommons.list_formatter(df.pregnancies))
        self.assertTrue(low <= np.mean(pregnancies) <= high)
