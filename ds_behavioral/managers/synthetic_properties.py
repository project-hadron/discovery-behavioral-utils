from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class DataBuilderPropertyManager(AbstractPropertyManager):
    """Class to deal with the properties of a data contract"""

    def reset_contract_properties(self):
        """resets the data contract properties back to it's original state. It also resets the connector handler"""
        super()._reset_abstract_properties()
        self._create_property_structure()
        return

    def _create_property_structure(self):
        if not self.is_key(self.KEY.generator_key):
            self.set(self.KEY.generator_key, {})
        return
