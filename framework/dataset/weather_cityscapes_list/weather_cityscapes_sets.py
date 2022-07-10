import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

pandas_file = "framework/dataset/weather_cityscapes_list/metadata.json"

base_file = pd.read_json(pandas_file)


def get_split(column, trainset, valset, restrictions={}):
    """
    Splits the dataframe to more datasets depending on the column, source and targets provided
    column: str of the pandas object column name
    source: tupple indicating the values of the column specified
    targets: list of tupples indicating all the target sets
    """

    trainset = [tuple(pair) for pair in trainset]
    valset = [tuple(pair) for pair in valset]
    selector = pd.Series(np.full(len(base_file), True))
    for fcol, value in restrictions.items():
        selector &= base_file[fcol] == value
    filtered = base_file[selector]

    all_sets = set(valset)  # {i for t in valset for i in t} # unravel
    all_sets |= set(trainset)  # all distinct values of sets
    output = {"train": {}, "val": {}}
    for set_ in all_sets:  # gathering the sets
        tmp = filtered[filtered[column].isin(set_)]
        if set_ in trainset:
            output["train"][set_] = tmp[tmp["set"] == "train"]
        if set_ in valset:
            output["val"][set_] = tmp[tmp["set"] == "val"]
    return output


def sample_split(validation=0.2, seed=12, upper_bound=True):
    source_class = [(0,)]
    targets = [[100]]
    return get_split("intensity", source_class, targets)


if __name__ == "__main__":
    print(sample_split())
