HarmonicNum = 5
MusicNet_Instruments = ["Piano", "Harpsichord", "Violin", "Viola", "Cello", "Contrabass", 
                        "Horn", "Oboe", "Bassoon", "Clarinet", "Flute"]
MusicNetMIDIMapping = {
    "Piano": 1,
    "Harpsichord": 7,
    "Violin": 41,
    "Viola": 42,
    "Cello": 43,
    "Contrabass": 44,
    "Horn": 61,
    "Oboe": 69,
    "Bassoon": 71,
    "Clarinet": 72,
    "Flute": 74
}
HarmonicNum = 5

def get_MusicNet_label_num_mapping(offset=1, spec_inst=None):
    ori_num = [1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74]
    
    mapping = {}

    if spec_inst is None:
        return mapping
    if "All" in spec_inst:
        spec_inst = MusicNet_Instruments

    for idx, name in enumerate(spec_inst):
        ii = MusicNet_Instruments.index(name)
        n = ori_num[ii]
        mapping[n] = idx + offset
    
    return mapping

def get_instruments_num(insts):
    if insts is None:
        instruments = 1
    elif "All" in insts:
        instruments = 11 # For MusicNet
    else:
        instruments = len(insts)
    
    return instruments

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
    base_path="./../data/MAPS/"
    label_ext=".txt"
    '''
    train_wavs=[
        "MAPS_AkPnBcht_2/AkPnBcht/MUS",
        "MAPS_AkPnBsdf_2/AkPnBsdf/MUS",
        "MAPS_AkPnStgb_2/AkPnStgb/MUS",
        "MAPS_AkPnCGdD_2/AkPnCGdD/MUS",
        "MAPS_SptkBGCl_2/SptKBGCl/MUS",
        "MAPS_StbgTGd2_2/StbgTGd2/MUS"
    ]
    test_wavs= [
        "MAPS_ENSTDkAm_2/ENSTDkAm/MUS",
        "MAPS_ENSTDkCl_2/ENSTDkCl/MUS"
    ]
    '''
    train_wavs=[
        "AkPnBcht/MUS/",
        "AkPnCGdD/MUS/",
        "AkPnBsdf/MUS/",
        "AkPnStgb/MUS/",
        "StbgTGd2/MUS/",
        "SptKBGCl/MUS/"
    ]
    test_wavs= [
        "ENSTDkAm/MUS/",
    ]
    train_labels=train_wavs
    test_labels=test_wavs