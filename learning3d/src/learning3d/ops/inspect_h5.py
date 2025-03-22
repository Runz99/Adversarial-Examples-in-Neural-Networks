import os
import h5py

def inspect_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File: {file_path}")
            print("Datasets in file:", list(f.keys()))
            if 'data' in f:
                print("Data shape:", f['data'].shape)
            if 'label' in f:
                print("Labels shape:", f['label'].shape)
    except Exception as e:
        print(f"Error inspecting file {file_path}: {e}")

# Inspect the training and testing files
inspect_h5_file(os.path.expanduser('~/fyp/finaldataset/modelnet40_train.h5'))
inspect_h5_file(os.path.expanduser('~/fyp/finaldataset/modelnet40_test.h5'))