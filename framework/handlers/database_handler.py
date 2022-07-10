import numpy as np
from framework.utils.serialization import json_load

DATABASE_NAMES = [
    "rainy_cityscapes_video",
    "external_video",
    "rainy_cityscapes",
    "fog_cityscapes",
]


def get_db(cfg):
    assert (
        cfg.SCHEME.DATASET in DATABASE_NAMES
    ), f"cfg.SCHEME.DATASET not in {DATABASE_NAMES}"
    src_list = cfg.SCHEME.SOURCE
    domains = list(src_list)
    domains.extend(cfg.SCHEME.DOMAIN_ORDER)
    domains_unique = np.unique(domains).tolist()
    dbs = []
    if cfg.SCHEME.DATASET == "rainy_cityscapes_video":
        from ..dataset.weather_cityscapes_list.weather_cityscapes_video_sets import (
            get_split as video_split,
        )

        json_obj = json_load("framework/dataset/cityscapes_list/info.json")
        json_obj["classnum_to_label"] = dict(
            zip(list(range(len(json_obj["label"]))), json_obj["label"])
        )
        for domain in domains:
            dbs.append(video_split(cfg.SCHEME.COLUMN, [domain], cfg.SCHEME.FILTERS))
    elif cfg.SCHEME.DATASET == "external_video":
        from ..dataset.bern_video.video_sets import get_split as external_video_split

        json_obj = json_load("framework/dataset/cityscapes_list/info.json")
        json_obj["classnum_to_label"] = dict(
            zip(list(range(len(json_obj["label"]))), json_obj["label"])
        )
        for domain in domains:
            dbs.append(
                external_video_split(cfg.SCHEME.COLUMN, [domain], cfg.SCHEME.FILTERS)
            )
    elif cfg.SCHEME.DATASET == "rainy_cityscapes":
        from ..dataset.weather_cityscapes_list.weather_cityscapes_sets import (
            get_split as rain_split,
        )

        json_obj = json_load("framework/dataset/cityscapes_list/info.json")
        json_obj["classnum_to_label"] = dict(
            zip(list(range(len(json_obj["label"]))), json_obj["label"])
        )
        for domain in domains:
            dbs.append(
                rain_split(cfg.SCHEME.COLUMN, [domain], [domain], cfg.SCHEME.FILTERS)
            )
    elif cfg.SCHEME.DATASET == "fog_cityscapes":
        from ..dataset.weather_cityscapes_list.weather_cityscapes_fog_sets import (
            get_split as fog_split,
        )

        json_obj = json_load("framework/dataset/cityscapes_list/info.json")
        json_obj["classnum_to_label"] = dict(
            zip(list(range(len(json_obj["label"]))), json_obj["label"])
        )
        for domain in domains:
            dbs.append(
                fog_split(cfg.SCHEME.COLUMN, [domain], [domain], cfg.SCHEME.FILTERS)
            )
    return {
        "domains_src": dbs[: len(src_list)],
        "domains_trg": dbs[len(src_list) :],
        "db_info": json_obj,
    }
