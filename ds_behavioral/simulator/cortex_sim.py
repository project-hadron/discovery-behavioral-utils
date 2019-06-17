from ds_discovery import Transition

__author__ = "Darryl Oatridge"


class CortexTransitionAgent(object):

    def __init__(self, identifier: str):
        name = "cortex_tr_{}".format(identifier)
        self._tr = Transition(name)

    def set_source_contract(self, source_contract: dict):
        self._tr.remove_source_contract()
        self._tr.data_pm.set(self._tr.data_pm.KEY.source_key, source_contract)
        self._tr.persist()

    def set_transition_pipeline(self, contract_pipeline: dict):
        self._tr.remove_cleaner()
        self._tr.data_pm.set(self._tr.data_pm.KEY.cleaners_key, contract_pipeline)
        self._tr.persist()

    def set_augmented_knowledge(self, augmented_knowledge: dict):
        self._tr.remove_notes()
        self._tr.data_pm.set(self._tr.data_pm.KEY.augmented_key, augmented_knowledge)
        self._tr.persist()

    def run_transition_pipeline(self):
        self._tr.remove_clean_canonical()
        return self._tr.refresh_clean_canonical()

    def source_canonical(self):
        return self._tr.load_source_canonical()

    def transitioned_canonical(self):
        return self._tr.load_clean_canonical()

    def report_source(self):
        return self._tr.report_source()

    def report_transition(self):
        return self._tr.report_cleaners()

    def report_notes(self):
        return self._tr.report_notes()

