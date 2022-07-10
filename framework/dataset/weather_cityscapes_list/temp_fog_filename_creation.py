from curses import meta
from importlib_metadata import metadata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_json("metadata.json")
only_train = df[(df["set"] == "train") & (df["intensity"] == 0)]
train_index, val_index = train_test_split(
    only_train.index, test_size=500, random_state=0
)
only_train.loc[val_index, "set"] = "val"


def path_for_fog(intensity):
    return only_train["image_path"].str.replace("/clear/", f"/fog/{intensity}/")


fog_intensities = ["150m", "30m", "375m", "40m", "50m", "750m", "75m"]
set_of_dfs = [only_train.copy()]
for intensity in fog_intensities:
    new_df = only_train.copy()
    new_df["image_path"] = path_for_fog(intensity)
    new_df["intensity"] = int(intensity[:-1])
    set_of_dfs.append(new_df.copy())
complete_db = pd.concat(set_of_dfs, ignore_index=True)
complete_db.to_json("metadata_fog_small_test.json")
