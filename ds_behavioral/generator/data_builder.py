import os
from ds_foundation.aistac.abstract_component import AbstractComponent
from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_behavioral.generator.data_builder_tools import DataBuilderTools
from ds_behavioral.managers.synthetic_properties import DataBuilderPropertyManager

__author__ = 'Darryl Oatridge'


class DataBuilderComponent(AbstractComponent):

    @classmethod
    def from_uri(cls, task_name: str, properties_uri: str, default_save=None):
        """ Class Factory Method that builds the connector handlers for the properties contract. The method uses
        the schema of the URI to determine if it is remote or local. s3:// schema denotes remote, empty schema denotes
        local.
        Note: the 'properties_uri' only provides a URI up to and including the path but not the properties file names.

         :param task_name: The reference name of the properties contract
         :param properties_uri: A URI that identifies the resource path. The syntax should be either
                          s3://<bucket>/<path>/ for remote or <path> for local
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        _uri = properties_uri
        if not isinstance(_uri, str) or len(_uri) == 0:
            raise ValueError("the URI must take the form 's3://<bucket>/<path>/' for remote or '<path>/' for local")
        _schema, _, _path = ConnectorContract.parse_address_elements(uri=_uri)
        if str(_schema).lower().startswith('s3'):
            connector_contract = cls._from_remote(task_name=task_name, pm_path_uri=_uri)
        else:
            _uri = _path
            connector_contract =  cls._from_local(task_name=task_name, pm_path_uri=_uri)
        intent_model = DataBuilderTools()
        pm = DataBuilderPropertyManager.from_properties(task_name, connector_contract)
        return cls(property_manager=pm, intent_model=intent_model, default_save=default_save)

    @classmethod
    def _from_remote(cls, task_name: str, pm_path_uri: str,) -> ConnectorContract:
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store.
        Note: the 'properties_uri' only provides a URI up to and including the path but not the properties file names.

         :param task_name: The reference name of the properties contract
         :param pm_path_uri: A URI that identifies the S3 properties resource path. The syntax should be:
                          s3://<bucket>/<path>/
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """

        _module_name = 'ds_connectors.handlers.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        _address = ConnectorContract.parse_address(uri=pm_path_uri)
        _query_kw = ConnectorContract.parse_query(uri=pm_path_uri)
        _data_uri = os.path.join(_address, "config_transition_data_{}.pickle".format(task_name))
        return ConnectorContract(uri=_data_uri, module_name=_module_name, handler=_handler, **_query_kw)

    @classmethod
    def _from_local(cls, task_name: str, pm_path_uri: str) -> ConnectorContract:
        """ Class Factory Method that builds the connector handlers from a local resource path.
        This assumes the use of the pandas handler module and yaml persisted file.

        :param task_name: The reference name of the properties contract
        :param pm_path_uri: (optional) A URI that identifies the properties resource path.
                            by default is '/tmp/aistac/contracts'
        :param default_save: (optional) if the configuration should be persisted
        :return: the initialised class instance
        """
        _module_name = 'ds_connectors.handlers.pandas_handlers'
        _handler = 'PandasPersistHandler'
        _data_uri = os.path.join(pm_path_uri, "config_transition_data_{}.yaml".format(task_name))
        return ConnectorContract(uri=_data_uri, module_name=_module_name, handler=_handler)

