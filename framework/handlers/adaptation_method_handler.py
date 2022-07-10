ADAPTATION_METHOD_NAMES = [
    "PROTO_ONLINE",
    "ADVENT",
    "PROTO_ONLINE_VSWITCH",
    "PROTO_ONLINE_HSWITCH",
    "PROTO_ADVENT",
    "PROTO_ONLINE_HYBRIDSWITCH",
]


def get_adapt_method(cfg):
    assert (
        cfg.METHOD.ADAPTATION.NAME in ADAPTATION_METHOD_NAMES
    ), f"cfg.METHOD.ADAPTATION.NAME not in {ADAPTATION_METHOD_NAMES}"
    if cfg.METHOD.ADAPTATION.NAME == "PROTO_ONLINE":
        from ..domain_adaptation.methods.prototypes import online_proDA

        return online_proDA
    elif cfg.METHOD.ADAPTATION.NAME == "ADVENT":
        from ..domain_adaptation.methods.advent_da import advent

        return advent
    elif cfg.METHOD.ADAPTATION.NAME == "PROTO_ONLINE_VSWITCH":
        from ..domain_adaptation.methods.prototypes_vswitch import vswitch_proDA

        return vswitch_proDA
    elif cfg.METHOD.ADAPTATION.NAME == "PROTO_ONLINE_HSWITCH":
        from ..domain_adaptation.methods.prototypes_hswitch import hswitch_proDA

        return hswitch_proDA
    elif cfg.METHOD.ADAPTATION.NAME == "PROTO_ADVENT":
        from ..domain_adaptation.methods.prototype_advent import adv_proDA

        return adv_proDA
    elif cfg.METHOD.ADAPTATION.NAME == "PROTO_ONLINE_HYBRIDSWITCH":
        from ..domain_adaptation.methods.prototypes_hybrid_switch import (
            hybrid_proDA,
        )

        return hybrid_proDA
