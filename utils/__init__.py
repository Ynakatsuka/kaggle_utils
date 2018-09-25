import functools
import gc
import time
import numpy as np
from IPython.display import display, Javascript
from .sampling import *
from .load import *


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('elapsed time: {:.1f} s'.format(end - start))
        return result
    return wrapper


def check_path(path):
    if os.path.exists(path):
        raise ValueError('{} already exists.'.format(path))
      
    
def save_notebook(current_ipynb_path, save_ipynb_path):
    print('Saving notebook...')
    display(Javascript("IPython.notebook.save_notebook()"), include=['application/javascript'])
    time.sleep(3)
    command = 'jupyter nbconvert --to notebook {} --output {}'.format(current_ipynb_path, save_ipynb_path)
    print('executing {}...'.format(command))
    os.system(command)
    print('Done.')   
    

# def submit(competition_name, submit_path, message=''):
#     command = 'kaggle competitions submit -c {} -f {} -m {}'.format(competition_name, submit_path, message)
#     os.system(command)


def submit(competition_name, file_path, comment='from API'):
    os.system(f'kaggle competitions submit -c {competition_name} -f {file_path} -m "{comment}"')
    time.sleep(60)
    tmp = os.popen(f'kaggle competitions submissions -c {COMPETITION_NAME} -v | head -n 2').read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += f'{i}: {j}\n'
#        print(f'{i}: {j}') # TODO: comment out later?
#     send_line(message.rstrip())


def reduce_dataframe_memory(dataframe, columns=None):
    start_mem = dataframe.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    if columns is None:
        columns = dataframe.columns

    for col in columns:
        col_type = dataframe[col].dtype

        if col_type != object:
            c_min = dataframe[col].min()
            c_max = dataframe[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataframe[col] = dataframe[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)
        else:
            dataframe[col] = dataframe[col].astype('category')
    gc.collect()

    end_mem = dataframe.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return dataframe
