import os
import h5py
import numpy as np

def combine_h5_files(input_dir, output_file, split):
    all_data = []
    all_labels = []

    # Iterate over all class files
    for class_idx, class_name in enumerate(sorted(os.listdir(input_dir))):
        if not class_name.endswith('.h5'):
            continue

        class_file = os.path.join(input_dir, class_name)
        with h5py.File(class_file, 'r') as f:
            print(f"Processing file: {class_file}")  # Debugging: Print the file being processed
            data = f['data'][:]
            labels = f['label'][:]

            # Debugging: Print the shape of the data
            print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

            # Skip files with incorrect data shape
            if len(data.shape) != 3 or data.shape[1] != 1024 or data.shape[2] != 3:
                print(f"Skipping file {class_file}: Expected shape (num_samples, 1024, 3), got {data.shape}")
                continue

            # Filter data based on the split (train/test)
            num_samples = data.shape[0]
            split_idx = int(num_samples * 0.8)  # 80% train, 20% test
            if split == 'train':
                data = data[:split_idx]
                labels = labels[:split_idx]
            elif split == 'test':
                data = data[split_idx:]
                labels = labels[split_idx:]

            all_data.append(data)
            all_labels.append(labels)

    # Debugging: Print the shapes of all_data arrays
    for i, arr in enumerate(all_data):
        print(f"Array {i} shape: {arr.shape}")

    # Concatenate all data and labels
    if all_data:  # Check if there are any valid arrays
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        print("No valid data found. Skipping file creation.")
        return

    # Save to output file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=all_data)
        f.create_dataset('label', data=all_labels)

if __name__ == '__main__':
    input_dir = os.path.expanduser('~/fyp/datasets')  # Directory with individual .h5 files
    output_train = os.path.expanduser('~/fyp/finaldataset/modelnet40_train.h5')  # Output training file
    output_test = os.path.expanduser('~/fyp/finaldataset/modelnet40_test.h5')  # Output testing file

    # Combine files for training and testing
    combine_h5_files(input_dir, output_train, split='train')
    combine_h5_files(input_dir, output_test, split='test')

    print(f"Combined training data saved to: {output_train}")
    print(f"Combined testing data saved to: {output_test}")