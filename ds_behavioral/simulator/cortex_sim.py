import pandas as pd

from ds_discovery import Transition

__author__ = "Darryl Oatridge"


class CortexTransition(object):

    def __init__(self):
        self._tr = Transition('cortex')

    def set_source(self, source_contract: dict):
        self._tr.data_pm.set(self._tr.data_pm.KEY.source_key, source_contract)

    def set_intent(self, contract_pipeline: dict):
        self._tr.data_pm.set(self._tr.data_pm.KEY.cleaners_key, contract_pipeline)
        self._tr.save()

    def run_intent_pipeline(self):
        return self._tr.refresh_clean_canonical()

    def load_source_canonical(self) -> pd.DataFrame:
        return self._tr.load_source_canonical()

    def load_clean_canonical(self) -> pd.DataFrame:
        return self._tr.load_clean_canonical()

    @property
    def source(self):
        return self._tr.data_pm.source

    @property
    def contract_pipeline(self):
        return self._tr.data_pm.cleaners

