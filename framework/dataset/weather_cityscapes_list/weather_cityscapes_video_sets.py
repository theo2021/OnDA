import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

pandas_file = "framework/dataset/weather_cityscapes_list/metadata_video.json"

base_file = pd.read_json(pandas_file)


def get_split(column, trainset, restrictions={}):
    """
    Splits the dataframe to more datasets depending on the column, source and targets provided
    column: str of the pandas object column name
    source: tupple indicating the values of the column specified
    targets: list of tupples indicating all the target sets
    """

    trainset = [tuple(pair) for pair in trainset]
    selector = pd.Series(np.full(len(base_file), True))
    for fcol, value in restrictions.items():
        selector &= base_file[fcol] == value
    filtered = base_file[selector]

    all_sets = set(trainset)  # all distinct values of sets
    output = {"train": {}}
    for set_ in all_sets:  # gathering the sets
        tmp = filtered[filtered[column].isin(set_)]
        if set_ in trainset:
            output["train"][set_] = tmp
    return output


def sample_split():
    source_class = [("source",)]
    targets = [["100mm_1"]]
    return get_split("scene", source_class + targets)


if __name__ == "__main__":
    print(sample_split())
