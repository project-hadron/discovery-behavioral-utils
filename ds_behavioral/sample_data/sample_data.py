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
    def _get_dataset(filename: str, size: int = None, seed: int = None) -> list:
        """private method to retrieve a dataset"""
        _path = Path(AbstractSample._full_path(filename))
        df = pd.read_csv(_path, header=None)
        return AbstractSample._select_list(selection=df.iloc[:, 0].tolist(), size=size, seed=seed)

    @staticmethod
    def _select_list(selection: list, size: int = None, seed: int = None):
        """private method to select from a series"""
        seed = int(time.time() * np.random.random()) if not isinstance(seed, int) else seed
        np.random.seed(seed)
        np.random.shuffle(selection)
        if not isinstance(size, int) or not 0 < size < len(selection):
            size = len(selection) - 1
        return selection[:size]


class ProfileSample(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(ProfileSample):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def female_names(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of female first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return ProfileSample._get_dataset(filename='lookup_female_first_names.csv', size=size, seed=seed)

    @staticmethod
    def male_names(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of male first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return ProfileSample._get_dataset(filename='lookup_male_first_names.csv', size=size, seed=seed)

    @staticmethod
    def surnames(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of surnames first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return ProfileSample._get_dataset(filename='lookup_last_names.csv', size=size, seed=seed)

    @staticmethod
    def professions(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of professions first names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return ProfileSample._get_dataset(filename='lookup_professions.csv', size=size, seed=seed)

    @staticmethod
    def uk_cities(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return CallCentreSamples._get_dataset(filename='lookup_uk_city.csv', size=size, seed=seed)

    @staticmethod
    def uk_postcode_district(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return CallCentreSamples._get_dataset(filename='lookup_uk_postcode_district.csv', size=size, seed=seed)

    @staticmethod
    def us_cities(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return CallCentreSamples._get_dataset(filename='lookup_us_city.csv', size=size, seed=seed)

    @staticmethod
    def us_zipcodes(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return CallCentreSamples._get_dataset(filename='lookup_us_zip.csv', size=size, seed=seed)

    @staticmethod
    def us_states(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             'AA', 'AE', 'AK', 'AL', 'AP', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'FM', 'GA', 'GU',
             'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MH', 'MI', 'MN', 'MO', 'MP', 'MS',
             'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'PW', 'RI', 'SC',
             'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)

    @staticmethod
    def global_mail_domains(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = pd.Series[
             "aol.com", "att.net", "comcast.net", "facebook.com", "gmail.com", "gmx.com", "googlemail.com",
             "google.com", "hotmail.com", "mac.com", "me.com", "mail.com", "msn.com", "live.com", "sbcglobal.net",
             "verizon.net", "yahoo.com", "email.com", "fastmail.fm", "games.com", "gmx.net", "hush.com", "hushmail.com",
             "icloud.com", "iname.com", "inbox.com", "lavabit.com", "love.com" "outlook.com", "pobox.com",
             "protonmail.com", "rocketmail.com", "safe-mail.net", "wow.com", "ygm.com", "ymail.com", "zoho.com",
             "yandex.com", "bellsouth.net", "charter.net", "cox.net", "earthlink.net", "juno.com"]
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)

    @staticmethod
    def british_mail_domains(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             "btinternet.com", "virginmedia.com", "blueyonder.co.uk", "freeserve.co.uk", "live.co.uk", "ntlworld.com",
             "o2.co.uk", "orange.net", "sky.com", "talktalk.co.uk", "tiscali.co.uk", "virgin.net", "wanadoo.co.uk",
             "bt.com", "yahoo.co.uk", "hotmail.co.uk"]
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)


class CallCentreSamples(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(CallCentreSamples):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def phrases(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return CallCentreSamples._get_dataset(filename='lookup_catch_phrases.csv', size=size, seed=seed)

    @staticmethod
    def slogans(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return CallCentreSamples._get_dataset(filename='lookup_slogan_phrases.csv', size=size, seed=seed)

    @staticmethod
    def contact_type(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             'Phone Call', 'E-mail', 'Letter', 'Internet', 'MyPortal', 'Questionnaire', 'Account manager',
             'E-mail & Phone Call', 'Letter & Phone Call', 'Visit', 'Fax', 'Retail Voice', 'Third Party Call',
             'Survey']
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)

    @staticmethod
    def complaint(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return ProfileSample._get_dataset(filename='lookup_complaints.csv', size=size, seed=seed)


class GenericSamples(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(GenericSamples):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def real_company_names(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of real company names of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return GenericSamples._get_dataset(filename='lookup_company_names.csv', size=size, seed=seed)

    @staticmethod
    def fake_company_names(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        return GenericSamples._get_dataset(filename='lookup_fake_company_names.csv', size=size, seed=seed)

    @staticmethod
    def road_types(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             'Alley', 'Avenue', 'Boulevard', 'Close', 'Circle', 'Crescent', 'Crossing', 'Court', 'Drive', 'Hill',
             'Lane', 'Road', 'Park', 'Parkway', 'Plaza', 'Street', 'Terrace', 'Way']
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)


class MutualFundSamples(AbstractSample):

    def __dir__(self):
        rtn_list = []
        for m in dir(MutualFundSamples):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def mutual_fund_type(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = ['Money market', 'Fixed income', 'Equity', 'Balanced', 'Index', 'Specialty', 'Fund-of-funds']
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)

    @staticmethod
    def pension_product(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             'Individual Pension', 'Annuity', 'Bond', 'Uncategorised', 'Savings', 'Term Assurance', 'Income Drawdown',
             'Freestanding AVC', 'Mortgage Protection', 'Corporate Pension']
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)

    @staticmethod
    def authority_type(size: int = None, seed: int = None) -> list:
        """returns a randomly selected list of size

        :param size: (optional) the size of the sample. If None then all the names are returned
        :param seed: (optional) a seed value
        :return: a list of names
        """
        selection = [
             'Policy Holder', 'Financial Adviser', '3rd Party Claims Reviewer', 'Relative of Policyholder',
             'Other Third Party', 'Executor', 'Trustee', 'Employer Contact', 'Life Assured', 'General Public']
        return ProfileSample._select_list(selection=selection, size=size, seed=seed)
