import pandas as pd
from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_behavioral.intent.synthetic_intent_model import SyntheticIntentModel
from ds_foundation.components.abstract_component import AbstractComponent
from ds_behavioral.managers.synthetic_property_manager import SyntheticPropertyManager

__author__ = 'Darryl Oatridge'


class SyntheticBuilder(AbstractComponent):

    CONNECTOR_SYNTHETIC = 'synthetic'

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, pm_file_type: str=None, default_save=None, **kwargs):
        """ Class Factory Method to instantiates the component application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.

        by default the handler is local Pandas but also supports remote AWS S3 and Redis. It use these Factory
        instantiations ensure that the schema is s3:// or redis:// and the handler will be automatically redirected

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param kwargs: to pass to the connector contract
         :return: the initialised class instance
         """
        _pm = SyntheticPropertyManager(task_name=task_name)
        _intent_model = SyntheticIntentModel(property_manager=_pm)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, pm_file_type=pm_file_type, **kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save)

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

    def set_outcome_path(self, path: str, file_type: str=None, version: str=None, save: bool=None):
        """ Sets the outcome persist contract using the PandasPersistHandler

        :param path: A path to place the synthetic outcome
        :param file_type: (optional) a file type supported by the PandasPersistHandler
        :param version: (optional) a version number to give the connector used in the output name
        :param save: (optional) if True, save to file. Default is True
        """
        file_type = file_type if isinstance(file_type, str) else 'csv'
        versioned = True if isinstance(version, str) else False
        uri = self.pm.file_pattern(path=path, connector_name=self.CONNECTOR_SYNTHETIC, file_type=file_type,
                                   versioned=versioned)
        self.set_outcome_uri(uri=uri, save=save)
        return

    def set_outcome_uri(self, uri: str, save: bool=None):
        """ Sets the outcome persist contract using the PandasPersistHandler

        :param uri: the URI for the synthetic outcome
        :param version: (optional) a version number to give the connector used in the output name
        :param save: (optional) if True, save to file. Default is True
        """
        outcome_contract = ConnectorContract(uri=uri, module_name="ds_connectors.handlers.pandas_handlers",
                                             handler="PandasPersistHandler")
        self.set_outcome_contract(outcome_contract=outcome_contract, save=save)
        return

    def set_outcome_contract(self, outcome_contract: ConnectorContract, save: bool=None):
        """ Sets the outcome persist contract

        :param outcome_contract: the connector contract for the synthetic outcome
        :param save: (optional) if True, save to file. Default is True
        """
        save = save if isinstance(save, bool) else self._default_save
        if self.pm.has_connector(self.CONNECTOR_SYNTHETIC):
            self.remove_connector_contract(self.CONNECTOR_SYNTHETIC)
        self.add_connector_contract(self.CONNECTOR_SYNTHETIC, connector_contract=outcome_contract, save=save)
        self.pm_persist(save)
        return

    def save_synthetic_canonical(self, df):
        """Saves the pandas.DataFrame to the clean files folder"""
        self.persist_canonical(self.CONNECTOR_SYNTHETIC, df)

    def report_connectors(self, connector_filter: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_connectors(connector_filter=connector_filter), orient='columns')
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['connector_name'], **{'font-weight': 'bold'})
            return df_style
        else:
            df.set_index(keys='connector_name', inplace=True)
        return df

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book(), orient='columns')
        if stylise:
            index = df[df['name'].duplicated()].index.to_list()
            df.loc[index, 'name'] = ''
            df = df.reset_index(drop=True)
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['name'],  **{'font-weight': 'bold', 'font-size': "120%"})
            return df_style
        return df

    def report_intent(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(), orient='columns')
        if stylise:
            index = df[df['level'].duplicated()].index.to_list()
            df.loc[index, 'level'] = ''
            df = df.reset_index(drop=True)
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['level'],  **{'font-weight': 'bold', 'font-size': "120%"})
            return df_style
        return df

    def report_notes(self, catalog: [str, list]=None, labels: [str, list]=None, regex: [str, list]=None,
                     re_ignore_case: bool=False, stylise: bool=True, drop_dates: bool=False):
        """ generates a report on the notes

        :param catalog: (optional) the catalog to filter on
        :param labels: (optional) s label or list of labels to filter on
        :param regex: (optional) a regular expression on the notes
        :param re_ignore_case: (optional) if the regular expression should be case sensitive
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param drop_dates: (optional) excludes the 'date' column from the report
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        drop_dates = False if not isinstance(drop_dates, bool) else drop_dates
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        report = self.pm.report_notes(catalog=catalog, labels=labels, regex=regex, re_ignore_case=re_ignore_case,
                                      drop_dates=drop_dates)
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['section'], **{'font-weight': 'bold'})
            _ = df_style.set_properties(subset=['label', 'section'], **{'font-size': "120%"})
            return df_style
        return df
