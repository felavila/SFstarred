import h5py


def read_hdf5_to_dict(file_path):
    if file_path.endswith(".hdf5") == False:
        return print("maybe the file is not a h5py")
    data_dict = {}
    filename =file_path
    with h5py.File(filename, 'r') as f:
        # Recursively traverse the HDF5 file
        def traverse(item, path=''):
            if isinstance(item, h5py.Dataset):
                data_dict[path] = item[()]
            elif isinstance(item, h5py.Group):
                for key in item.keys():
                    traverse(item[key], path+ key)
        traverse(f)
    return data_dict
    
def save_dict_as_h5py(my_dict,file_path):
    if file_path.endswith(".hdf5") == False:
        file_path = file_path + ".hdf5"
    # Open the HDF5 file in write mode
    with h5py.File(file_path, 'w') as f:
        # Iterate over the dictionary items and save each one as a dataset
        for key, value in my_dict.items():
            f.create_dataset(key, data=value)