from os.path import abspath, join, dirname
import time
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from aistac.handlers.abstract_handlers import HandlerFactory

__author__ = 'Darryl Oatridge'


class AbstractSample(ABC):

    @staticmethod
    def _full_path(filename):
        return abspath(join(dirname(__file__), filename))

    @abstractmethod
    def __dir__(self):
        pass

    @staticmethod
    def _get_constant(reference: str, size: int=None, shuffle: bool=True, seed: int=None) -> [pd.DataFrame, list]:
        """private method to retrieve data constant"""
        module = HandlerFactory.get_module(module_name=f"ds_behavioral.sample.{reference}")
        if reference.startswith("lookup_"):
            return AbstractSample._select_list(selection=module.data, size=size, seed=seed, shuffle=shuffle)
        df = pd.DataFrame.from_dict(module.data, orient='columns')
        idx = df.index.to_list()
        selection = AbstractSample._select_list(selection=idx, size=size, seed=seed, shuffle=shuffle)
        rtn_df: pd.DataFrame = df.iloc[selection].reset_index(drop=True)
        return rtn_df

    @staticmethod
    def _get_dataset(filename: str, size: int=None, shuffle: bool=True, seed: int=None, header: bool=True):
        """private method to retrieve a dataset"""
        header = 'infer' if header else None
        _path = Path(AbstractSample._full_path(filename))
        df = pd.read_csv(_path, encoding='latin1', header=header)
        idx = df.index.to_list()
        df = df.iloc[AbstractSample._select_list(selection=idx, size=size, seed=seed, shuffle=shuffle)]
        return df if df.shape[1] > 1 else df.iloc[:, 0].to_list()

    @staticmethod
    def _select_list(selection: list, size: int=None, shuffle: bool=True, seed: int=None):
        """private method to select from a series
        :param shuffle:
        """
        size = size if isinstance(size, int) else len(selection)
        seed = seed if isinstance(seed, int) else int(time.time() * np.random.random())
        np.random.seed(seed)
        selection *= int(((size - 1) / len(selection)) + 1)
        if shuffle:
            idx = np.random.randint(0, len(selection), size=size)
            return pd.Series(selection).iloc[idx].values.tolist()[:size]
        return selection[:size]


class MappedSample(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(MappedSample):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def us_healthcare_organisations(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_healthcare_organisations', size=size, seed=seed,
                                            shuffle=shuffle)

    @staticmethod
    def us_healthcare_practitioner(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        # The PCP tax code-
        seed = int(time.time() * np.random.random()) if not isinstance(seed, int) else seed
        np.random.seed(seed)
        df = AbstractSample._get_constant(reference='map_us_real_addresses', shuffle=False)
        num_choice = np.linspace(100000000, 900000000, num=300000, dtype=int, endpoint=False)
        num_choice += np.random.randint(100, 999, size=300000)
        df['pcp_tax_id'] = list(np.random.choice(num_choice, size=df.shape[0], replace=False))

    @staticmethod
    def companies_fortune1000(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_companies_fortune1000', size=size, seed=seed,
                                            shuffle=shuffle)

    @staticmethod
    def companies_inc5000(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_companies_inc5000', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_profession_rank(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_profession_detail_rank', size=size, seed=seed,
                                            shuffle=shuffle)

    @staticmethod
    def uk_postcodes_primary(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_uk_postcodes_primary', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_city_area_code(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        # return AbstractSample._get_constant(reference='map_us_city_area_code', size=size, seed=seed, shuffle=shuffle)
        return AbstractSample._get_dataset(filename='map_us_city_area_code.csv', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_city_zipcodes_rank(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_city_zipcodes_rank', size=size, seed=seed,
                                            shuffle=shuffle)

    @staticmethod
    def us_zipcode_primary(size: int=None, cleaned: bool=False, shuffle: bool=True, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param cleaned: (optional) if all decommissioned and nan values should be removed
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        df = AbstractSample._get_constant(reference='map_us_zipcode_primary', size=size, seed=seed, shuffle=shuffle)
        if cleaned:
            df = df.dropna(subset=['State'])
        pop_total = df['EstimatedPopulation'].sum()
        df['WeightedPopulation'] = df['EstimatedPopulation'].apply(lambda x: np.round((x/pop_total) * 100000, 2))
        return df

    @staticmethod
    def us_phone_code(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_phone_code', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_surname_rank(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_surname_rank', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_forename_unisex(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_forename_unisex', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_full_address(size: int=None, shuffle: bool=False, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        return AbstractSample._get_constant(reference='map_us_full_address', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_forename_mf(female_bias: float=None, size: int=None, shuffle: bool=True, seed: int=None) -> pd.DataFrame:
        """returns the first 'size' dataframe where the female_bias is between zero and 1

        :param female_bias: a female bias between 0 and 1 where 0 is zero females and 1 is all females
        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: the mapping DataFrame
        """
        female_bias = female_bias if isinstance(female_bias, float) and 0 <= female_bias <= 1 else 0.5
        shuffle = shuffle if isinstance(shuffle, bool) else True
        size = size if isinstance(size, int) else 10000
        df = AbstractSample._get_constant(reference='map_us_forename_mf', shuffle=False)
        df.columns = ['forename', 'gender']
        # generate a binomial probability of female_bias
        generator = np.random.default_rng()
        female_bias = pd.Series(list(generator.binomial(n=1, p=female_bias, size=1000)))
        female_bias = np.round(female_bias.value_counts().loc[1]/1000, 2)
        female_size = int(np.round(female_bias*size, 0))
        female_idx = df[df['gender'] == 'F'].dropna().index.to_list()
        female_idx *= int(((female_size - 1) / len(female_idx)) + 1)
        male_size = size - female_size
        if male_size > 0:
            male_idx = df[df['gender'] == 'M'].dropna().index.to_list()
            male_idx *= int(((male_size - 1) / len(male_idx)) + 1)
        else:
            male_idx = []
        idx = female_idx[:female_size] + male_idx[:male_size]
        if shuffle:
            seed = int(time.time() * np.random.random()) if not isinstance(seed, int) else seed
            np.random.seed(seed)
            np.random.shuffle(idx)
        return df.iloc[idx].reset_index(drop=True)


class Sample(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(Sample):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def us_surnames(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of surnames weighted on popularity over 150,000 of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """

        def divider(_size):
            weight_map = {1000000: 50000, 500000: 100000, 150000: 500000}
            for k, v in weight_map.items():
                if _size > k:
                    return v
            return 2000000

        size = size if isinstance(size, int) else 150000
        df = AbstractSample._get_constant(reference='map_us_surname_rank', seed=seed, shuffle=False)
        df['weight'] = [int(round((x / divider(size)), 0)) for x in df['count']]
        df_clean = df.where(df.weight > 1).dropna()
        result = [[x] * df_clean.weight.astype(int).iloc[x] for x in df_clean.index]
        idx = [j for i in result for j in i] + df.index.to_list()
        idx = AbstractSample._select_list(selection=idx, size=size, shuffle=shuffle, seed=seed)
        return df['name'].iloc[idx].to_list()

    @staticmethod
    def us_professions(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of profession names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        df = AbstractSample._get_constant(reference='map_us_profession_detail_rank', seed=seed, shuffle=False)
        df.columns = ['occupation', 'total']
        size = size if isinstance(size, int) else df.shape[0]
        result = [[x] * df['total'].astype(int).iloc[x] for x in df.index]
        idx = [j for i in result for j in i]
        idx = AbstractSample._select_list(selection=idx, size=size, shuffle=shuffle, seed=seed)
        return df['occupation'].iloc[idx].to_list()

    @staticmethod
    def uk_street_types(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = ['Road', 'Street', 'Way', 'Avenue', 'Drive', 'Lane', 'Grove', 'Gardens', 'Place', 'Circus',
                     'Crescent', 'Bypass', 'Close', 'Square', 'Hill', 'Mews', 'Vale', 'Rise', 'Row', 'Mead', 'Wharf']
        return Sample._select_list(selection=selection, size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def uk_cities(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_uk_city', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def uk_postcode_district(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_uk_postcode_district', size=size, seed=seed,
                                            shuffle=shuffle)

    @staticmethod
    def us_street_names(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_us_street_names', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_street_types(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_us_street_suffix', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_cities(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_us_city', size=size, seed=seed, shuffle=shuffle)

    @staticmethod
    def us_zipcodes(size: int = None, shuffle: bool=True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_us_zipcode', size=size, seed=seed, shuffle=shuffle)

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
    def complaint(size: int = None, shuffle: bool = True, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param shuffle: (optional) if the list should be shuffled. Default is True
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return AbstractSample._get_constant(reference='lookup_complaints', size=size, seed=seed, shuffle=shuffle)
