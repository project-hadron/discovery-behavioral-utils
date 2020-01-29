from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class SyntheticPropertyManager(AbstractPropertyManager):
    """property manager for the Synthetic Data Builder"""

    def __init__(self, task_name: str):
        """initialises the properties manager.

        :param task_name: the name of the task name within the property manager
        """
        super().__init__(task_name, root_keys=[], knowledge_keys=['describe'])


    @classmethod
    def manager_name(cls) -> str:
        return str(cls.__name__).lower().replace('propertymanager', '')
