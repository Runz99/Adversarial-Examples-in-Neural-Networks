import h5py
import numpy as np

h5_file = "/home/trkz99/fyp/finaldataset/modelnet40_train.h5"

with h5py.File(h5_file, "r") as f:
    labels = f["label"][:]
    print("Unique labels in H5 file:", np.unique(labels))


