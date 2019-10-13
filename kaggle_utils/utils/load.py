import glob
import pickle
import numpy as np
import pandas as pd



def check_columns(df, columns):
    non_duplicated = list(set(df.columns) - set(columns))
    duplicated = list(set(df.columns) & set(columns))
    columns.extend(non_duplicated)
    return df[non_duplicated], columns, duplicated
    
    
def load_features(file_names, data_dir='../data/working/features/', print_path=False, print_duplicated=False):
    features = []
    columns = []
    for file_name in file_names:
        duplicated = []
        load_paths = glob.glob(data_dir+file_name+'.*')
        if (len(load_paths) > 1) or (len(load_paths) == 0):
            raise ValueError('Invalid file name. Found path: ', load_paths)
        if print_path:
            print('Loading {}...'.format(load_paths[0]))
        extension = load_paths[0].split('.')[-1]
        if extension == 'ftr':
            df, columns, duplicated = check_columns(pd.read_feather(load_paths[0]), columns)
            features.append(df)
        elif extension == 'pkl':
            df, columns, duplicated = check_columns(pd.read_pickle(load_paths[0]), columns)
            features.append(df)
        elif extension == 'npy':
            feature = np.load(load_paths[0])
            if len(feature.shape) == 1:
                npy_columns = [file_name]
            else:
                npy_columns = [file_name + str(i) for i in range(feature.shape[1])]
            df = pd.DataFrame(feature, columns=npy_columns)
            df, columns, duplicated = check_columns(df, columns)
            features.append(df)
        else:
            raise ValueError('invalid extension: ', extension, ' of ', load_paths[0])
        if len(duplicated) & print_duplicated:
            print('duplicated in ', load_paths[0])
            print('columns: ', duplicated)

    return pd.concat(features, axis=1)


def load_list_features(file_names, data_dir='../data/working/features/', load_type='list'):
    if load_type == 'list':
        features = []
    elif load_type == 'dict':
        features = {}
    else:
        raise ValueError('invalid load_type')
        
    for file_name in file_names:
        load_paths = glob.glob(data_dir+file_name+'.*')
        if (len(load_paths) > 1) or (len(load_paths) == 0):
            raise ValueError('Invalid file name. Found path: ', load_paths)
        print('Loading {}...'.format(load_paths[0]))
        extension = load_paths[0].split('.')[-1]
        if extension == 'pkl':
            with open(load_paths[0] , mode='rb') as f:
                tmp = pickle.load(f)
            if load_type == 'list':
                features.extend(tmp)
            elif load_type == 'dict':
                features.update(tmp)
        else:
            raise ValueError('invalid extension: ', extension)

    return features
