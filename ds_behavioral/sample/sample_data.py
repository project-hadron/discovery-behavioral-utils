from os.path import abspath, join, dirname
import time
from pathlib import Path

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

__author__ = 'Darryl Oatridge'


class AbstractSample(ABC):

    @staticmethod
    def _full_path(filename):
        return abspath(join(dirname(__file__), filename))

    @abstractmethod
    def __dir__(self):
        pass

    @staticmethod
    def _get_dataset(filename: str, size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """private method to retrieve a dataset"""
        _path = Path(AbstractSample._full_path(filename))
        df = pd.read_csv(_path, header=None)
        return AbstractSample._select_list(selection=df.iloc[:, 0].tolist(), size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def _select_list(selection: list, size: int = None, shuffle: bool=True, seed: int = None):
        """private method to select from a series
        :param shuffle:
        """
        seed = int(time.time() * np.random.random()) if not isinstance(seed, int) else seed
        np.random.seed(seed)
        if shuffle:
            np.random.shuffle(selection)
        if not isinstance(size, int) or not 0 < size < len(selection):
            size = len(selection) - 1
        return selection[:size]


class MappedSample(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(MappedSample):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def companies_fortune1000(size: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('map_companies_fortune1000.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        return df.iloc[:size]

    @staticmethod
    def companies_inc5000(size: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('map_companies_inc5000.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        return df.iloc[:size]

    @staticmethod
    def uk_postcodes_primary(size: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('map_uk_postcodes_primary.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        return df.iloc[:size]

    @staticmethod
    def us_zipcode_primary(size: int=None, cleaned: bool=False) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param cleaned: (optional) if all decommissioned and nan values should be removed
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('map_us_zipcode_primary.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        if cleaned:
            df = df.dropna(subset=['State'])
        pop_total = df['EstimatedPopulation'].sum()
        df['WeightedPopulation'] = df['EstimatedPopulation'].apply(lambda x: np.round((x/pop_total) * 100000, 2))
        return df.iloc[:size]

    @staticmethod
    def profile_us_500(size: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('profile_us_500.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        return df.iloc[:size]

    @staticmethod
    def profile_uk_500(size: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('profile_uk_500.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        return df.iloc[:size]

    @staticmethod
    def profile_au_500(size: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :return: the mapping DataFrame
        """
        _path = Path(AbstractSample._full_path('profile_au_500.csv'))
        df = pd.read_csv(_path, encoding='latin1')
        return df.iloc[:size]


class Sample(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(Sample):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def names(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of names taken from other sample such as company, surname etc.
        Note: These are title case and might have spaces

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = Sample.surnames(seed=seed) + Sample.uk_cities(seed=seed)
        selection += Sample.us_cities(seed=seed) + Sample.company_names(seed=seed)
        return pd.Series(selection).str.title().to_list()

    @staticmethod
    def female_names(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of female first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_female_first_names.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def male_names(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of male first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_male_first_names.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def surnames(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of surnames first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_last_names.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def professions(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of professions first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_professions.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def uk_cities(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_uk_city.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def uk_postcode_district(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_uk_postcode_district.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def us_cities(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_us_city.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_zipcodes(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_us_zipcode.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_states(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             'AA', 'AE', 'AK', 'AL', 'AP', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'FM', 'GA', 'GU',
             'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MH', 'MI', 'MN', 'MO', 'MP', 'MS',
             'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'PW', 'RI', 'SC',
             'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def global_mail_domains(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             "hotmail.com", "google.com", "facebook.com", "gmail.com", "googlemail.com", "msn.com", "verizon.net",
             "yahoo.com", "aol.com", "att.net", "comcast.net", "gmx.com", "mac.com", "me.com", "mail.com", "live.com",
             "sbcglobal.net", "email.com", "fastmail.fm", "games.com", "gmx.net", "hush.com", "hushmail.com",
             "icloud.com", "iname.com", "inbox.com", "lavabit.com", "love.com" "outlook.com", "pobox.com",
             "protonmail.com", "rocketmail.com", "safe-mail.net", "wow.com", "ygm.com", "ymail.com", "zoho.com",
             "yandex.com", "bellsouth.net", "charter.net", "cox.net", "earthlink.net", "juno.com"]
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def british_mail_domains(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             "btinternet.com", "virginmedia.com", "blueyonder.co.uk", "freeserve.co.uk", "live.co.uk", "ntlworld.com",
             "o2.co.uk", "orange.net", "sky.com", "talktalk.co.uk", "tiscali.co.uk", "virgin.net", "wanadoo.co.uk",
             "bt.com", "yahoo.co.uk", "hotmail.co.uk"]
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def company_fortune_1000(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of real company names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_fortune1000_companies.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def company_names(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_inc5000_companies.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def slogan_mechanic(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_slogan_mechanics.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def contact_type(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
            'Phone Call', 'E-mail', 'Letter', 'Internet', 'MyPortal', 'Questionnaire', 'Account manager',
            'E-mail & Phone Call', 'Letter & Phone Call', 'Visit', 'Fax', 'Retail Voice', 'Third Party Call',
            'Survey']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def complaint(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_complaints.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def phrases(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_catch_phrases.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def slogans(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return Sample._get_dataset(filename='lookup_slogan_phrases.csv', size=size, seed=seed,
                                   shuffle=shuffle)

    @staticmethod
    def road_types(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
            'Alley', 'Avenue', 'Boulevard', 'Close', 'Circle', 'Crescent', 'Crossing', 'Court', 'Drive', 'Hill',
            'Lane', 'Road', 'Park', 'Parkway', 'Plaza', 'Street', 'Terrace', 'Way']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def mutual_fund_type(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = ['Money market', 'Fixed income', 'Equity', 'Balanced', 'Index', 'Specialty', 'Fund-of-funds']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def pension_product(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
            'Individual Pension', 'Annuity', 'Bond', 'Uncategorised', 'Savings', 'Term Assurance', 'Income Drawdown',
            'Freestanding AVC', 'Mortgage Protection', 'Corporate Pension']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def authority_type(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
            'Policy Holder', 'Financial Adviser', '3rd Party Claims Reviewer', 'Relative of Policyholder',
            'Other Third Party', 'Executor', 'Trustee', 'Employer Contact', 'Life Assured', 'General Public']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)
