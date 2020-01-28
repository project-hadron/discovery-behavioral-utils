from ds_foundation.components.abstract_component import AbstractComponent
from ds_behavioral.managers.data_builder_property_manager import DataBuilderPropertyManager
from ds_behavioral.generator.data_builder_tools import DataBuilderTools

__author__ = 'Darryl Oatridge'


class DataBuilderComponent(AbstractComponent):

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, default_save=None, **kwargs):
        _pm = DataBuilderPropertyManager(task_name=task_name, root_keys=[], knowledge_keys=[])
        _intent_model = DataBuilderTools(property_manager=_pm)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, **kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save)

    @property
    def tools(self) -> DataBuilderTools:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> DataBuilderPropertyManager:
        """The properties manager instance"""
        return self._component_pm
