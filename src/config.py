HarmonicNum = 5

class BaseDatasetInfo:
    base_path = ""
    label_ext=""
    train_wavs=""
    test_wavs=""
    train_labels=""
    test_labels=""

    def __init__(self, base_path):
        self.base_path = base_path

class MAPSDatasetInfo(BaseDatasetInfo):
    base_path="./../data/MAPS"
    label_ext=".txt"
    train_wavs=[
        #"MAPS_AkPnBcht_2/AkPnBcht/MUS",
        #"MAPS_AkPnBsdf_2/AkPnBsdf/MUS",
        #"MAPS_AkPnStgb_2/AkPnStgb/MUS",
        "MAPS_AkPnCGdD_2/AkPnCGdD/MUS",
        #"MAPS_SptkBGCl_2/SptKBGCl/MUS",
        #"MAPS_StbgTGd2_2/StbgTGd2/MUS"
    ]
    test_wavs= [
        "MAPS_ENSTDkAm_2/ENSTDkAm/MUS",
        #"MAPS_ENSTDkCl_2/ENSTDkCl/MUS"
    ]
    train_labels=train_wavs
    test_labels=test_wavs