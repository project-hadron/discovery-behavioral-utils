import pandas as pd
from aistac.handlers.abstract_handlers import ConnectorContract
from ds_behavioral.components.commons import SyntheticCommons
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from aistac.components.abstract_component import AbstractComponent
from ds_behavioral.managers.synthetic_property_manager import SyntheticPropertyManager

__author__ = 'Darryl Oatridge'


class SyntheticBuilder(AbstractComponent):

    CONNECTOR_OUTCOME = 'outcome'

    def __init__(self, property_manager: SyntheticPropertyManager, intent_model: SyntheticIntentModel,
                 default_save=None, reset_templates: bool = None, align_connectors: bool = None):
        """ Encapsulation class for the discovery set of classes

        :param property_manager: The contract property manager instance for this components
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
        :param align_connectors: (optional) resets aligned connectors to the template
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, align_connectors: bool=None, default_save_intent: bool=None,
                 default_intent_level: bool=None, order_next_available: bool=None, default_replace_intent: bool=None,
                 has_contract: bool=None):
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param username: A user name for this task activity.
         :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param pm_module: (optional) the module or package name where the handler can be found
         :param pm_handler: (optional) the handler for retrieving the resource
         :param pm_kwargs: (optional) a dictionary of kwargs to pass to the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :param has_contract: (optional) indicates the instance should have a property manager domain contract
         :return: the initialised class instance
         """
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_connectors.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        _pm = SyntheticPropertyManager(task_name=task_name, username=username)
        _intent_model = SyntheticIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                             default_intent_level=default_intent_level,
                                             order_next_available=order_next_available,
                                             default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def _from_remote_s3(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store."""
        _module_name = 'ds_connectors.handlers.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        return _module_name, _handler

    @property
    def pm(self) -> SyntheticPropertyManager:
        return self._component_pm

    @property
    def intent_model(self) -> SyntheticIntentModel:
        return self._intent_model

    @property
    def tools(self) -> SyntheticIntentModel:
        return self._intent_model

    def get_outcome_contract(self):
        """ gets the outcome connector contract that can be used as the next chain source"""
        return self.pm.get_connector_contract(self.CONNECTOR_OUTCOME)

    def set_outcome_contract(self, outcome_contract: ConnectorContract, save: bool=None):
        """ Sets the outcome persist contract

        :param outcome_contract: the connector contract for the synthetic outcome
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(self.CONNECTOR_OUTCOME, connector_contract=outcome_contract, save=save)
        return

    def set_outcome(self, uri_file: str=None, save: bool=None, **kwargs):
        """sets the outcome contract CONNECTOR_OUTCOME using the TEMPLATE_PERSIST connector contract

        :param uri_file: the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        file_pattern = self.pm.file_pattern(connector_name=self.CONNECTOR_OUTCOME)
        uri_file = uri_file if isinstance(uri_file, str) else file_pattern
        self.add_connector_from_template(connector_name=self.CONNECTOR_OUTCOME, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)

    def load_synthetic_canonical(self) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        return self.load_canonical(self.CONNECTOR_OUTCOME)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        return canonical

    def save_synthetic_canonical(self, canonical):
        """Saves the pandas.DataFrame to the clean files folder"""
        self.persist_canonical(connector_name=self.CONNECTOR_OUTCOME, canonical=canonical)

    def add_column_description(self, column_name: str, description: str, save: bool=None):
        """ adds a description note that is included in with the 'report_column_catalog'"""
        if isinstance(description, str) and description:
            self.pm.set_intent_description(level=column_name, text=description)
            self.pm_persist(save)
        return

    def run_synthetic_pipeline(self, size: int, columns: [str, list]=None):
        """Runs the transition pipeline from source to persist"""
        result = self.intent_model.run_intent_pipeline(size=size, columns=columns)
        self.save_synthetic_canonical(canonical=result)

    def report_connectors(self, connector_filter: [str, list] = None, stylise: bool = True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_connectors(connector_filter=connector_filter), orient='columns')
        if stylise:
            SyntheticCommons.report(df, index_header='connector_name')
        df.set_index(keys='connector_name', inplace=True)
        return df

    def report_run_book(self, stylise: bool = True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book(), orient='columns')
        if stylise:
            SyntheticCommons.report(df, index_header='name')
        df.set_index(keys='name', inplace=True)
        return df

    def report_intent(self, stylise: bool = True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(), orient='columns')
        if stylise:
            SyntheticCommons.report(df, index_header='level')
        df.set_index(keys='level', inplace=True)
        return df

    def report_notes(self, catalog: [str, list] = None, labels: [str, list] = None, regex: [str, list] = None,
                     re_ignore_case: bool = False, stylise: bool = True, drop_dates: bool = False):
        """ generates a report on the notes

        :param catalog: (optional) the catalog to filter on
        :param labels: (optional) s label or list of labels to filter on
        :param regex: (optional) a regular expression on the notes
        :param re_ignore_case: (optional) if the regular expression should be case sensitive
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param drop_dates: (optional) excludes the 'date' column from the report
        :return: pd.Dataframe
        """
        report = self.pm.report_notes(catalog=catalog, labels=labels, regex=regex, re_ignore_case=re_ignore_case,
                                      drop_dates=drop_dates)
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            SyntheticCommons.report(df, index_header='section', bold='label')
        df.set_index(keys='section', inplace=True)
        return df

    def report_column_catalog(self, column_name: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param column_name: (optional) filters on specific column names.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=column_name, as_description=True,
                                                               level_label='column_name'), orient='columns')
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['column_name'], **{'font-weight': 'bold'})
            return df_style
        else:
            df.set_index(keys='column_name', inplace=True)
        return df

