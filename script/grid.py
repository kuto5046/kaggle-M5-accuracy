import gc
import os
import sys
import time
import json
import pickle
import psutil
import random
import warnings
import traceback
import numpy as np
import pandas as pd
from math import ceil
from sklearn.preprocessing import LabelEncoder


## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def read_data(index_columns, END_TRAIN, TARGET):
    print("\nRead data ->")
    # Here are reafing all our data 
    # without any limitations and dtype modification
    train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
    prices_df = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
    calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
    grid_df = pd.melt(train_df, id_vars = index_columns, var_name = 'd', value_name = TARGET)
    print('Train rows:', len(train_df), len(grid_df))

    # add stage1 data(d1914 ~ d1941)
    add_grid = pd.DataFrame()
    for i in range(1,29):
        temp_df = train_df[index_columns]
        temp_df = temp_df.drop_duplicates()
        temp_df['d'] = 'd_'+ str(END_TRAIN+i)
        temp_df[TARGET] = np.nan
        add_grid = pd.concat([add_grid, temp_df])

    grid_df = pd.concat([grid_df,add_grid])
    grid_df = grid_df.reset_index(drop=True)

    # Remove
    del temp_df, add_grid, train_df

    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    # We can free some memory 
    # by converting "strings" to categorical
    # it will not affect merging and 
    # we will not lose any valuable data
    for col in index_columns:
        grid_df[col] = grid_df[col].astype('category')

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    return grid_df, prices_df, calendar_df


# create data which include releace info(what's day product was releaced)
def make_base_grid(grid_df, prices_df, calendar_df):
    """
    find the release week and remove the unreleased period.
    it we will have not very accurate release week 

    """
    print('\nRelease week')

    # wm_yr_wk形式でreleace weekを特定
    release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id','item_id','release']
    grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
    del release_df

    # wm_yr_wkをcalendarから持ってくる
    grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
                        
    # リリースしていない列は削除
    grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
    grid_df = grid_df.reset_index(drop=True)

    # Let's check our memory usage
    print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    # releaceが一番若い日(ほぼd1)を基準としてreleace日を算出 
    grid_df['release'] = grid_df['release'] - grid_df['release'].min()
    grid_df['release'] = grid_df['release'].astype(np.int16)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    
    return grid_df


def make_prices_grid(grid_df, prices_df, calendar_df):

    # We can do some basic aggregations
    prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

    # Some items are can be inflation dependent
    # and some items are very "stable"
    prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

    # I would like some "rolling" aggregations
    # but would like months and years as "window"
    calendar_prices = calendar_df[['wm_yr_wk','month','year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
    del calendar_prices

    # Now we can add price "momentum" (some sort of)
    # Shifted by week 
    # by month mean
    # by year mean
    prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

    del prices_df['month'], prices_df['year']

    # Merge Prices
    original_columns = list(grid_df)
    grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
    keep_columns = [col for col in list(grid_df) if col not in original_columns]
    grid_df = grid_df[['id','d'] + keep_columns]
    grid_df = reduce_mem_usage(grid_df)
    
    del prices_df

    return grid_df


def make_calendar_grid(grid_df, calendar_df):

    grid_df = grid_df[['id','d']]

    # Merge calendar partly
    icols = ['date',
            'd',
            'event_name_1',
            'event_type_1',
            'event_name_2',
            'event_type_2',
            'snap_CA',
            'snap_TX',
            'snap_WI']

    grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

    # Minify data
    # 'snap_' columns we can convert to bool or int8
    icols = ['event_name_1',
            'event_type_1',
            'event_name_2',
            'event_type_2',
            'snap_CA',
            'snap_TX',
            'snap_WI']
    for col in icols:
        grid_df[col] = grid_df[col].astype('category')

    # Convert to DateTime
    grid_df['date'] = pd.to_datetime(grid_df['date'])

    # Make some features from date
    grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
    grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
    grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
    grid_df['tm_y'] = grid_df['date'].dt.year
    grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
    grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

    grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
    grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)

    # Remove date
    del grid_df['date'], calendar_df

    return grid_df


def grid_cleaning(grid_path):
    # Convert 'd' to int
    grid_df = pd.read_pickle(grid_path + 'grid_part_1.pkl')
    grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

    # Remove 'wm_yr_wk'
    # as test values are not in train set
    del grid_df['wm_yr_wk']
    grid_df.to_pickle(grid_path + 'grid_part_1.pkl')


def summary(grid_path):
    # Now we have 3 sets of features
    grid_df = pd.concat([pd.read_pickle(grid_path + 'grid_part_1.pkl'),
                         pd.read_pickle(grid_path + 'grid_part_2.pkl').iloc[:,2:],
                         pd.read_pickle(grid_path + 'grid_part_3.pkl').iloc[:,2:]],
                         axis=1)

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    print('Size:', grid_df.shape)

    # 2.5GiB + is is still too big to train our model
    # (on kaggle with its memory limits)
    # and we don't have lag features yet
    # But what if we can train by state_id or shop_id?
    state_id = 'CA'
    grid_df = grid_df[grid_df['state_id']==state_id]
    print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    #           Full Grid:   1.2GiB

    store_id = 'CA_1'
    grid_df = grid_df[grid_df['store_id']==store_id]
    print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
    #           Full Grid: 321.2MiB


def main():
    # var
    TARGET = 'sales'
    END_TRAIN = 1941         # Last day in train set
    index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

    grid_path = "../input/m5-simple-fe/"
    os.makedirs(grid_path, exist_ok=True)

    grid_df, prices_df, calendar_df = read_data(index_columns, END_TRAIN, TARGET)
    
    # make and save grids 
    grid_df = make_base_grid(grid_df, prices_df, calendar_df)
    print('\nSave Part 1')
    grid_df.to_pickle(grid_path + 'grid_part_1.pkl')
    print('Size:', grid_df.shape)

    grid_df = make_prices_grid(grid_df, prices_df, calendar_df)
    print('\nMerege preices and save part 2')
    grid_df.to_pickle(grid_path + 'grid_part_2.pkl')
    print('Size:', grid_df.shape)

    grid_df = make_calendar_grid(grid_df, calendar_df)
    print('\nSave part 3')
    grid_df.to_pickle(grid_path + 'grid_part_3.pkl')
    print('Size:', grid_df.shape)
    
    # del grid_df
    
    # Some additional cleaning
    grid_cleaning(grid_path)

    # summary the data
    summary(grid_path)


if __name__ == "__main__":
    main()














