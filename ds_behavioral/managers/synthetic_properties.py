from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class DataBuilderPropertyManager(AbstractPropertyManager):
    """Class to deal with the properties of a data contract"""

    @classmethod
    def from_properties(cls, contract_name: str, connector_contract: ConnectorContract, replace: bool=True):
        """ A Factory initialisation method to load the parameters from persistance at instantiation

        :param contract_name: the name of the contract or subset within the property manager
        :param connector_contract: the SourceContract bean for the SourcePersistHandler
        :param replace: (optional) if the loaded properties should replace any in memory
        """
        manager = 'aistac_synthetic'
        instance = cls(property_manager=manager, task_name=contract_name, root_keys=[], knowledge_keys=[])
        instance.set_property_connector(connector_contract=connector_contract)
        if instance.get_connector_handler(instance.CONNECTOR_PM_CONTRACT).exists():
            instance.load_properties(replace=False)
        return instance

    def reset_contract_properties(self):
        """resets the data contract properties back to it's original state. It also resets the connector handler
        Note: this method ONLY writes to the properties memmory and must be explicitely persisted
        using the ``save()'' method
        """
        super()._reset_abstract_properties()
        self._create_property_structure()
        return

    def _create_property_structure(self):
        if not self.is_key(self.KEY.generator_key):
            self.set(self.KEY.generator_key, {})
        return
