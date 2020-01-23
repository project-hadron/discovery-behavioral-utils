from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class DataBuilderPropertyManager(AbstractPropertyManager):
    """property manager for the Synthetic Data Builder"""

    @classmethod
    def manager_name(cls) -> str:
        return str(cls.__name__).lower().replace('propertymanager', '')
