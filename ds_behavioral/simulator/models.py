import pandas as pd
from ds_behavioral import DataBuilderTools

__author__ = "Darryl Oatridge"


class Models(object):

    @staticmethod
    def recommend_heuristic(profile: pd.Series, items: pd.DataFrame, recommend: int=None, top: int=None,
                            exclude_items: list=None) -> list:
        """ takes a profile of an entity where the index is the coresponds to the


        :param profile:
        :param entities:
        :param items:
        :param recommend:
        :param top:
        :param exclude_items:
        :return:
        """
        recommend = 5 if recommend is None else recommend
        top = 3 if top is None else top
        # drop the entities in the exclude
        items.drop(exclude_items, axis=0, inplace=True)
        if profile is None:
            return []
        categories = profile.sort_values(ascending=False).iloc[:top]
        choice = DataBuilderTools.get_category(selection=categories.index.to_list(),
                                               weight_pattern=categories.values.tolist(), size=recommend)
        item_select = dict()
        for col in categories.index:
            item_select.update({col: items[col].sort_values(ascending=False).iloc[:recommend].index.to_list()})
        rtn_list = []
        for item_choice in choice:
            rtn_list.append(item_select.get(item_choice).pop())
        return rtn_list

