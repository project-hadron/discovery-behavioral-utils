import pandas as pd
from ds_behavioral import SyntheticBuilder

__author__ = "Darryl Oatridge"


class Models(object):

    @staticmethod
    def recommend_heuristic(profile: pd.Series, items: pd.DataFrame, recommend: int=None, top: int=None,
                            exclude_items: list=None) -> list:
        """ takes a profile of an entity where the index of the profile represents the columns in the items.
        for example the profile will be an index list or film genres and how many times these categories
        have been watched. The items will be columns of categories with the index the films and row values
        being the count of film watches in the column categories

        :param profile: a pandas series of categories (index) counters for a single profile
        :param items: a pandas dataframe of item counts (index) of columns (categories
        :param recommend: the number of recommended items to select from
        :param top: limits the cut-off of the top categories to select from
        :param exclude_items: item index to not include
        :return: a list of recommendations
        """
        recommend = 10 if recommend is None else recommend
        top = 10 if top is None or top < 1 else top
        # drop the entities in the exclude
        _df = items.drop(index=exclude_items, errors='ignore')
        if profile is None or profile.size == 0:
            return []
        categories = profile.sort_values(ascending=False).iloc[:top]
        choices = SyntheticBuilder.scratch_pad().get_category(selection=categories.index.to_list(),
                                                              relative_freq=categories.values.tolist(),
                                                              size=recommend)
        choices_count = pd.Series(choices).value_counts()
        selection_dict = {}
        for index in choices_count.index:
            selection_dict.update({index: _df[index].sort_values(ascending=False).iloc[
                                          :choices_count.loc[index]].index.to_list()})
        rtn_list = []
        for item in choices:
            rtn_list.append(selection_dict[item].pop())
        return rtn_list
