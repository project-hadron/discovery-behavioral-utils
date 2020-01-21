from ds_foundation.aistac.abstract_component import AbstractComponent
from ds_foundation.properties.abstract_properties import AbstractPropertyManager

from ds_behavioral.intent.data_builder_tools import DataBuilderTools

__author__ = 'Darryl Oatridge'


class DataBuilderComponent(AbstractComponent):

    @classmethod
    def _property_manager_class_name(cls) -> str:
        return AbstractPropertyManager.__name__

    @classmethod
    def _intent_model_class_name(cls) -> str:
        return DataBuilderTools.__name__