import os
import pandas as pd
from ds_foundation.managers.augment_properties import AugmentedPropertyManager
from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_foundation.properties.decorator_patterns import deprecated
from ds_behavioral.generator.data_bulder_tools import DataBuilderTools
from ds_behavioral.managers.synthetic_properties import DataBuilderPropertyManager

__author__ = 'Darryl Oatridge'


class DataBuilder(object):

    PERSIST_CONNECTOR = 'persist_connector'

    def __init__(self, contract_name: str, builder_properties: [ConnectorContract],
                 knowledge_properties: [ConnectorContract], default_save=None):
        """ Encapsulation class for the discovery set of classes

        :param contract_name: The name of the contract
        :param builder_properties: The persist handler for the builder properties
        :param knowledge_properties: The persist handler for the augmented knowledge properties
        :param default_save: The default behaviour of persisting the contracts:
                    if True: all contract properties are persisted
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        """
        if not isinstance(contract_name, str) or len(contract_name) < 1:
            raise ValueError("The contract name must be a valid string")
        self._contract_name = contract_name
        self._default_save = default_save if isinstance(default_save, bool) else True
        # set property managers
        self._builder_pm = DataBuilderPropertyManager.from_properties(contract_name=contract_name,
                                                                      connector_contract=builder_properties)
        if self._builder_pm.has_persisted_properties():
            self._builder_pm.load_properties()
        self._knowledge_catalogue = ['overview', 'notes', 'observations', 'attribute', 'dictionary', 'tor']
        self._knowledge_pm = AugmentedPropertyManager.from_properties(self._contract_name,
                                                                      connector_contract=knowledge_properties,
                                                                      knowledge_catalogue=self._knowledge_catalogue)
        if self._knowledge_pm.has_persisted_properties():
            self._knowledge_pm.load_properties()
        # initialise the values
        self.persist_properties(save=self._default_save)

        self._tools = DataBuilderTools()

    @classmethod
    def from_path(cls, contract_name: str,  contract_path: str, default_save=None):
        """ Class Factory Method that builds the connector handlers from the data paths.
        This assumes the use of the pandas handler module and yaml persisted file.

        :param contract_name: The reference name of the properties contract
        :param contract_path: (optional) the path of the properties contracts
        :param default_save: (optional) if the configuration should be persisted
        :return: the initialised class instance
        """
        for param in ['contract_name', 'contract_path']:
            if not isinstance(eval(param), str) or len(eval(param)) == 0:
                raise ValueError("a {} must be provided".format(param))
        _default_save = default_save if isinstance(default_save, bool) else True
        _module_name = 'ds_connectors.handlers.pandas_handlers'
        _location = os.path.join(contract_path, contract_name)
        _synthetic_connector = ConnectorContract(resource="config_synthetic_builder_{}.yaml".format(contract_name),
                                                 connector_type='yaml', location=_location, module_name=_module_name,
                                                 handler='PandasPersistHandler')
        _knowledge_connector = ConnectorContract(resource="config_augmented_{}.yaml".format(contract_name),
                                                 connector_type='yaml', location=_location, module_name=_module_name,
                                                 handler='PandasPersistHandler')
        return cls(contract_name=contract_name, builder_properties=_synthetic_connector,
                   knowledge_properties=_knowledge_connector, default_save=default_save)

    @classmethod
    def from_env(cls, contract_name: str,  default_save=None):
        """ Class Factory Method that builds the connector handlers taking the property contract path from
        either the os.envon['DBU_CONTRACT_PATH'], os.environ['DBU_PERSIST_PATH']/contracts or locally from the current
        working directory 'dbu/contracts' if no environment variable is found. This assumes the use of the default
        handler module and yaml persisted file.

         :param contract_name: The reference name of the properties contract
         :param default_save: (optional) if the configuration should be persisted
         :return: the initialised class instance
         """
        if 'SYNTHETIC_CONTRACT_PATH' in os.environ.keys():
            contract_path = os.environ['SYNTHETIC_CONTRACT_PATH']
        elif 'SYNTHETIC_PERSIST_PATH' in os.environ.keys():
            contract_path = os.path.join(os.environ['SYNTHETIC_PERSIST_PATH'], 'contracts')
        else:
            contract_path = os.path.join(os.getcwd(), 'synthetic', 'contracts')
        return cls.from_path(contract_name=contract_name, contract_path=contract_path, default_save=default_save)

    @property
    def contract_name(self) -> str:
        """The contract name of this transition instance"""
        return self._contract_name

    @property
    def version(self):
        """The version number of the contracts"""
        return self._builder_pm.version

    @property
    @deprecated("Data Builder method fpm has been deprecated as of version 1.02.006. Use builder_pm instead.")
    def fbpm(self) -> DataBuilderPropertyManager:
        """
        :return: the file builder properties instance
        """
        return self._builder_pm

    @property
    def builder_pm(self) -> DataBuilderPropertyManager:
        """
        :return: the file builder properties instance
        """
        return self._builder_pm

    @property
    def tools(self):
        """
        :return: the file builder tools instance
        """
        return self._tools

    @property
    def tool_dir(self):
        return self._tools.__dir__()

    def set_version(self, version, save=None):
        """ sets the version
        :param version: the version to be set
        :param save: if True, save to file. Default is True
        """
        if not isinstance(save, bool):
            save = self._default_save
            self._builder_pm.set_version(version=version)
        self.persist_properties(save)
        return

    def load_synthetic_data(self) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        if self.builder_pm.has_connector(self.PERSIST_CONNECTOR):
            handler = self.builder_pm.get_connector_handler(self.PERSIST_CONNECTOR)
            df = handler.load_canonical()
            return df
        return pd.DataFrame()

    def save_synthetic_data(self, df):
        """Saves the pandas.DataFrame to the clean files folder"""
        if self.builder_pm.has_connector(self.PERSIST_CONNECTOR):
            handler = self.builder_pm.get_connector_handler(self.PERSIST_CONNECTOR)
            handler.persist_canonical(df)
        return

    def remove_synthetic_data(self):
        """removes the current persisted canonical"""
        if self.builder_pm.has_connector(self.PERSIST_CONNECTOR):
            handler = self.builder_pm.get_connector_handler(self.PERSIST_CONNECTOR)
            handler.remove_canonical()
        return

    def persist_properties(self, save=None):
        """Saves the current configuration to file"""
        if not isinstance(save, bool):
            save = self._default_save
        if save:
            self._builder_pm.persist_properties()
            self._knowledge_pm.persist_properties()
        return

    def set_synthetic_persist_contract(self, resource=None, connector_type=None, location=None, module_name: str=None,
                                       handler: str=None, save: bool=None, **kwargs):
        """ Sets the persist contract for the synthetic data

        :param resource: a local file, connector, URI or URL
        :param connector_type: (optional) a reference to the type of resource. if None then csv file assumed
        :param location: (optional) a path, region or uri reference that can be used to identify location of resource
        :param module_name: a module name with full package path e.g 'ds_discovery.handlers.pandas_handlers
        :param handler: the name of the Handler Class. Must be
        :param save: if True, save to file. Default is True
        :param kwargs: (optional) a list of key additional word argument properties associated with the resource
        :return: if load is True, returns a Pandas.DataFrame else None
        """
        save = save if isinstance(save, bool) else self._default_save
        if resource is None or not resource:
            resource = self.get_persist_file_name('synthetic')
        if connector_type is None:
            connector_type = 'csv'
        if location is None:
            if 'SYNTHETIC_PERSIST_PATH' in os.environ.keys():
                location = os.environ['SYNTHETIC_PERSIST_PATH']
            else:
                raise ValueError("A location must be provided if the os.environ['SYNTHETIC_PERSIST_PATH'] is not set")
        module_name = 'ds_foundation.handlers.python_handlers' if module_name is None else module_name
        handler = 'PythonPersistHandler' if handler is None else handler
        self._builder_pm.set_connector_contract(self.PERSIST_CONNECTOR, resource=resource,
                                                connector_type=connector_type, location=location,
                                                module_name=module_name, handler=handler, **kwargs)
        self.persist_properties(save)
        return

    def get_persist_file_name(self, prefix: str):
        """ Returns a persist pattern based on name"""
        _pattern = "{}_{}_{}.csv"
        return _pattern.format(prefix, self.contract_name, self.version)
