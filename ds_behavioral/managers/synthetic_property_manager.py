from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.transition.commons import Commons

__author__ = 'Darryl Oatridge'


class SyntheticPropertyManager(AbstractPropertyManager):
    """property manager for the Synthetic Data Builder"""

    def __init__(self, task_name: str):
        """initialises the properties manager.

        :param task_name: the name of the task name within the property manager
        """
        super().__init__(task_name=task_name, root_keys=[], knowledge_keys=['describe'])

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
