from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class DataBuilderPropertyManager(AbstractPropertyManager):
    """Class to deal with the properties of a data contract"""
    MANAGER_NAME = 'synthetic'

    def __init__(self, contract_name):
        """ initialises the class specific to a data properties contract name

        :param contract_name: the contract reference name for these data properties
        """
        if contract_name is None or not isinstance(contract_name, str):
            assert ValueError("The contract_name can't be None or of zero length. '{}' passed".format(contract_name))
        keys = ['generator']
        super().__init__(manager=self.MANAGER_NAME, contract=contract_name, keys=keys)
        self._create_property_structure()

    @classmethod
    def from_properties(cls, contract_name: str, connector_contract: ConnectorContract, replace: bool=True):
        """ A Factory initialisation method to load the parameters from persistance at instantiation

        :param contract_name: the name of the contract or subset within the property manager
        :param connector_contract: the SourceContract bean for the SourcePersistHandler
        :param replace: (optional) if the loaded properties should replace any in memory
        """
        replace = replace if isinstance(replace, bool) else False
        instance = cls(contract_name=contract_name)
        instance.set_property_connector(resource=connector_contract.resource,
                                        connector_type=connector_contract.connector_type,
                                        location=connector_contract.location,
                                        module_name=connector_contract.module_name,
                                        handler=connector_contract.handler, **connector_contract.kwargs)
        if instance.get_connector_handler(instance.CONTRACT_CONNECTOR).exists():
            instance.load_properties(replace=replace)
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
