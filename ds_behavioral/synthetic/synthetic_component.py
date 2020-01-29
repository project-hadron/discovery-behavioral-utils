from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from ds_foundation.components.abstract_component import AbstractComponent
from ds_behavioral.managers.synthetic_property_manager import SyntheticPropertyManager

__author__ = 'Darryl Oatridge'


class SyntheticComponent(AbstractComponent):

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, module_name: str=None, handler: str=None,
                 default_save=None, **kwargs):
        _pm = SyntheticPropertyManager(task_name=task_name, root_keys=[], knowledge_keys=[])
        _intent_model = SyntheticIntentModel(property_manager=_pm)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, module_name=module_name,
                                 handler=handler, **kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save)

    @property
    def tools(self) -> SyntheticIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> SyntheticPropertyManager:
        """The properties manager instance"""
        return self._component_pm
