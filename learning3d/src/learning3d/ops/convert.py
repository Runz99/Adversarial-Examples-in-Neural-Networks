import os
import numpy as np
import h5py

def read_off(file_path):
    try:
        with open(file_path, 'r') as f:
            # Read the first line and check if it starts with 'OFF'
            first_line = f.readline().strip()
            if first_line.startswith('OFF'):
                # If the header and numbers are on the same line, split them
                if first_line != 'OFF':
                    parts = first_line.split()
                    if len(parts) >= 4:  # Ensure there are enough parts
                        header = parts[0]
                        n_verts = int(parts[1])
                        n_faces = int(parts[2])
                        # Skip the rest of the first line and proceed
                    else:
                        raise ValueError(f"Invalid header in file: {file_path}")
                else:
                    # Standard OFF format: read the next line for numbers
                    n_verts, n_faces, _ = map(int, f.readline().strip().split())
            else:
                raise ValueError(f"Not a valid OFF header in file: {file_path}")

            # Read the vertices
            verts = []
            for _ in range(n_verts):
                line = f.readline().strip()
                if line:
                    verts.append([float(x) for x in line.split()])
                else:
                    raise ValueError(f"Unexpected end of file: {file_path}")

            return np.array(verts)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None  # Skip this file

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    furthest_distance = np.max(np.sqrt(np.sum(pc**2, axis=-1)))
    pc = pc / furthest_distance
    return pc

def convert_off_to_h5(off_dir, h5_dir, num_points=1024):
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)

    classes = [d for d in os.listdir(off_dir) if os.path.isdir(os.path.join(off_dir, d))]
    classes.sort()

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(off_dir, class_name)
        h5_file = os.path.join(h5_dir, f'{class_name}.h5')

        with h5py.File(h5_file, 'w') as f:
            data = []
            labels = []

            for split in ['train', 'test']:
                split_dir = os.path.join(class_dir, split)
                for off_file in os.listdir(split_dir):
                    if off_file.endswith('.off') and not off_file.startswith('.'):  # Skip hidden files
                        off_path = os.path.join(split_dir, off_file)
                        if not os.path.exists(off_path):  # Verify the file exists
                            print(f"File not found: {off_path}")
                            continue
                        print(f"Processing file: {off_path}")  # Debugging: Print the file being processed
                        pc = read_off(off_path)
                        if pc is None:  # Skip invalid files
                            continue
                        pc = normalize_point_cloud(pc)
                        if pc.shape[0] > num_points:
                            pc = pc[:num_points, :]
                        elif pc.shape[0] < num_points:
                            pc = np.pad(pc, ((0, num_points - pc.shape[0]), (0, 0)), mode='constant')
                        data.append(pc)
                        labels.append(i)

            data = np.array(data, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            f.create_dataset('data', data=data)
            f.create_dataset('label', data=labels)

if __name__ == '__main__':
    off_dir = os.path.expanduser('~/datasets/ModelNet40')  # Replace with the path to your ModelNet40 .off files
    h5_dir = os.path.expanduser('~/fyp/datasets')  # Replace with the path where you want to save the .h5 files
    convert_off_to_h5(off_dir, h5_dir)