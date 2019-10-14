import datetime
from ds_behavioral.generator.data_builder_tools import DataBuilderTools

__author__ = "Darryl Oatridge"


class ClaimsSimulator(object):

    @staticmethod
    def get_claims_id(sample_size: int):
        """ returns a list of claims like identifiers that are unique each day

        :param sample_size: the size of the returned sample list.
        :return: a list of unique claim like identifiers
        """
        prefix = datetime.datetime.now().strftime("%Y%m%-d")
        chars = []
        numbers = []
        result = []
        for _ in range(sample_size + 500):
            if len(chars) == 0:
                selection = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
                chars = DataBuilderTools.unique_str_tokens(length=3, pool=selection, size=10000)
            if len(numbers) == 0:
                numbers = DataBuilderTools.unique_numbers(100000, 999999, size=99999)
            result.append(f'{prefix}{chars.pop()}{numbers.pop()}')

        return list(set(result))[:sample_size]
