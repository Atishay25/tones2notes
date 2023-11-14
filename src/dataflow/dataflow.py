import os
import h5py
import pickle
import numpy as np

from keras.utils import Sequence

class BaseDataflow(Sequence):
    structure = None

    def __init__(self, 
                dataset_path,
                label_conversion_func,
                phase, 
                train_val_split=0.8, 
                use_ram=False, 
                timesteps=128, 
                b_sz=32, 
                channels=[1, 3], 
                feature_num=384, 
                out_classes=3, 
                **kwargs):

        self.dataset_path    = dataset_path
        self.use_ram         = use_ram
        self.train_val_split = train_val_split
        self.timesteps       = timesteps
        self.b_sz            = b_sz
        self.channels        = channels
        self.feature_num     = feature_num
        self.out_classes     = out_classes
        self.phase           = phase
        self.l_conv_func     = label_conversion_func

        self.features, self.labels = self.load_data(phase, use_ram=use_ram)

        self.init_index()
        self.post_init(**kwargs)

    def post_init(self, **kwargs):
        pass

    def init_index(self):
        self.idxs = []
        for i, pi in enumerate(self.features):
            iid = [(i, a) for a in range(0, len(pi), self.timesteps)]
            self.idxs += iid
        
        if not self.phase == "test":
            np.random.shuffle(self.idxs)

        b_itv = len(self.idxs) // self.b_sz
        self.b_start_idx = [a for a in range(0, len(self.idxs), b_itv)]
        self.b_idx_len = np.array([b_itv for _ in range(self.b_sz)])
        
        diff = self.b_sz - len(self.b_start_idx)
        self.b_idx_len[:diff] += 1
        for i in range(diff):
            self.b_start_idx[i+1:] += 1

        self.b_offset = self.b_start_idx.copy()
        self.b_idx_len[-1] = len(self.idxs) - self.b_start_idx[-1]

        self.d_buffer = np.zeros((self.b_sz, self.timesteps, self.feature_num, len(self.channels)))
        self.l_buffer = np.zeros((self.b_sz, self.timesteps, self.feature_num, self.out_classes))
        self.batch = 0

    def __getitem__(self, i):
        while self.batch < self.b_sz:
            if self.phase=="test":
                # Warning!
                # We suggest to use EvalFlow class rather than original Dataflow class for test phase
                pid, tid = self.idxs[i]
                x, y = self.get_feature(pid, tid)
            else:
                ii = i % self.b_sz
                bi = self.b_start_idx[ii]
                b_off = self.b_offset[ii]
                self.b_offset[ii] = (b_off+1) % self.b_idx_len[ii]
                iid = (bi+b_off) % self.b_idx_len[ii]

                pid, tid = self.idxs[iid]
                x, y = self.get_feature(pid, tid)

            self.d_buffer[self.batch] = x
            self.l_buffer[self.batch] = y
            self.batch += 1

        #if self.batch == self.b_sz:
        self.batch = 0
        return self.d_buffer, self.l_buffer
        
    def __len__(self):
        # Total available steps
        if self.phase == "test":
            return len(self.idxs)
        return np.ceil(len(self.idxs)/self.b_sz).astype('int')
    
    def get_feature(self, pid, tid):
        h_ref = self.features[pid]
        cc = BaseDataflow
        x = cc.pad_hdf(h_ref, tid, self.channels, timesteps=self.timesteps, feature_num=self.feature_num)
        y = self.l_conv_func(self.labels[pid], tid)
        
        return x, y

    def load_data(self, phase, use_ram=False):
        raise NotImplementedError
    

    def parse_feature_labels(self, hdf_paths, use_ram=False):
        feature = []
        labels = []
        for path in hdf_paths:
            hdf = h5py.File(path, "r")
            label = pickle.load(open(path.replace(".hdf", ".pickle"), "rb"))
            for key, value in hdf.items():
                feature.append(value[:] if use_ram else value)
                labels.append(label[key])

        return feature, labels
    
    def parse_files(self, path, ext):
        paths = []
        for root, dirs, files in os.walk(path):
            for ff in files:
                if ff.endswith(ext):
                    paths.append(os.path.join(root, ff))
        print("Feature path : ",path)
        if len(paths) == 0:
            raise ValueError("There is no feature file in the given path: {}".format(path))

        return paths
    
    def parse_hdf(self, paths, use_ram=False):
        hdf_insts = []
        
        for pp in paths:
            f = h5py.File(pp, "r")
            for i in range(len(f)):
                inst = f[str(i)]
                hdf_insts.append(inst[:] if use_ram else inst)

        return hdf_insts
    
    def parse_pickle(self, paths):
        pkls = []
        
        for pp in paths:
            pkl = pickle.load(open(pp, "rb"))
            for value in pkl:
                pkls.append(value)
        
        return pkls
    
    def pad_label(self, labels):
        # Padding data to the special data structure of label for this project
        # labels should contain both training and validation label

        new_x = [{} for _ in range(self.timesteps)]
        for p in labels:
            new_x.append(p)
        for _ in range(self.timesteps):
            new_x.append({})
        
        return new_x
    
    @staticmethod
    def pad_hdf(piece, tid, channels, timesteps=128, feature_num=384):
        feature = np.zeros((timesteps, feature_num, len(channels)))

        assert(feature_num >= piece.shape[1])
        h = piece.shape[1]
        p_b = (feature_num - h) // 2
        insert_range = range(p_b, p_b+h)
        t_len = min(timesteps, len(piece)-tid)

        feature[:t_len, insert_range] = piece[tid:(tid+timesteps), :, channels]

        return feature
    

class MAPSDataflow(BaseDataflow):
    structure = {
        "train":       "train_feature",
        "train_label": "train_feature",
        "val":         "train_feature",
        "val_label":   "train_feature",
        "test":        "test_feature",
        "test_label":  "test_feature"
    }

    def load_data(self, phase, use_ram=False):
        """ 
        Create hdf/pickle instance to the data

        Args:
            phase: "train", "val", or se_ram: load data into ram
        Returns:
            feature: h5py references to the feature
            label:   specilaized data structure of lables
        """

        if self.structure is None:
            raise NotImplementedError
        
        if phase == "train":
            path = os.path.join(self.dataset_path, self.structure["train"])
            g_path = os.path.join(self.dataset_path, self.structure["train_label"])
        elif phase == "val":
            path = os.path.join(self.dataset_path, self.structure["val"])
            g_path = os.path.join(self.dataset_path, self.structure["val_label"])
        elif phase == "test":
            path = os.path.join(self.dataset_path, self.structure["test"])
            g_path = os.path.join(self.dataset_path, self.structure["test_label"])
        else:
            raise TypeError
        hdfs = self.parse_files(path, ".hdf")
        print(self.dataset_path, path)
        return self.parse_feature_labels(hdfs)

    def post_init(self, **kwargs):
        # Split train/val data from training set
        split_p = round(len(self.features) * self.train_val_split)
        if self.phase == "train":
            self.features = self.features[:split_p]
            self.labels = self.labels[:split_p]
        elif self.phase == "val":
            self.features = self.features[split_p:]
            self.labels = self.labels[split_p:]

        self.init_index()