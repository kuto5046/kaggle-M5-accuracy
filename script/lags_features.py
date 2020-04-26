import gc
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import psutil

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def main():
    # read grid data
    grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')
    grid_df = grid_df[['id','d','sales']]
    TARGET = "sales"
    SHIFT_DAY = 28

    # Lags with 28 day shift
    start_time = time.time()
    print('Create lags')

    LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]
    grid_df = grid_df.assign(**{
            '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
            for l in LAG_DAYS
            for col in [TARGET]
        })

    # Minify lag columns
    for col in list(grid_df):
        if 'lag' in col:
            grid_df[col] = grid_df[col].astype(np.float16)

    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    # Rollings
    # with 28 day shift
    start_time = time.time()
    print('Create rolling aggs')

    for i in [7,14,30,60,180]:
        print('Rolling period:', i)
        grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
        grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)

    # Rollings
    # with sliding shift
    for d_shift in [1,7,14]: 
        print('Shifting period:', d_shift)
        for d_window in [7,14,30,60]:
            col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)
            grid_df[col_name] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
        
    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    print('Save lags and rollings')
    lags_path = "../input/m5-lags-features/"
    os.makedirs(lags_path, exist_ok = True)
    grid_df.to_pickle(lags_path + 'lags_df_'+str(SHIFT_DAY)+'.pkl')


if __name__ == "__main__":
    main()
