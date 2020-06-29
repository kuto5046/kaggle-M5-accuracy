import gc
import json
import os
import pickle
import random
import time
from IPython.core.display import display
import traceback
import psutil
import warnings
from datetime import datetime, timedelta
from typing import Union
from multiprocessing import Pool 

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn import metrics, preprocessing
from tqdm import tqdm
from utils import seed_everything, send_slack_notification, send_slack_error_notification

# Read data every stores
def get_data_by_key_column(KEY_COLUMN, TARGET, START_TRAIN, key_id):

    #PATHS for Features
    BASE     = '../input/m5-simple-fe/grid_part_1_1941.pkl'
    PRICE    = '../input/m5-simple-fe/grid_part_2.pkl'
    CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
    LAGS     = '../input/m5-lags-features/lags_df_28.pkl'
    MEAN_ENC = '../input/m5-custom-features/mean_encoding_df.pkl'
    

    # FEATURES to remove
    # These features lead to overfit or values not present in test set     
    if KEY_COLUMN == "dept_store_id":
        remove_features = ['id','state_id', KEY_COLUMN, 'dept_id', 'store_id', 'date','wm_yr_wk', 'd', TARGET]
    else:    
        remove_features = ['id','state_id', KEY_COLUMN, 'dept_store_id', 'date','wm_yr_wk', 'd', TARGET]
        

    mean_features  = ['enc_cat_id_mean','enc_cat_id_std',
                    'enc_dept_id_mean','enc_dept_id_std',
                    'enc_item_id_mean','enc_item_id_std'] 

    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)

    # Leave only relevant store
    df = df[df[KEY_COLUMN]==key_id]

    # With memory limits we have to read 
    # lags and mean encoding features
    # separately and drop items that we don't need.
    # As our Features Grids are aligned 
    # we can use index to keep only necessary rows
    # Alignment is good for us as concat uses less memory than merge.
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    # df["d"] = df["d"].str.strip("d_").astype(int)

    # Create featur es list
    features = [col for col in list(df) if col not in remove_features]
    print("number of features: ", len(features))
    df = df[['id','d',TARGET] + features]
    
    # Skipping first n rows
    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    
    return df, features


def main(KEY_COLUMN):
    t1 = time.time()

    # var
    VER = 6                          # Our model version
    TARGET = "sales"
    # KEY_COLUMN = 'store_id'     # training each id
    NUM_CPU = psutil.cpu_count() 
    SEED = 5046                      # We want all things
    seed_everything(SEED)            # to be as deterministic 

    #LIMITS and const
    START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
    END_TRAIN   = 1941               # TODO 最終的に1941に変更 End day of our train set 
    P_HORIZON   = 28                 # Prediction horizon

    # NOW_DATE = datetime.today().strftime("%Y%m%d_%H%M%S")
    # ORIGINAL = '../input/m5-forecasting-accuracy/' 
    OUTPUT = '../output/{}/'.format("v" + str(VER) + "_" + KEY_COLUMN + "_" + str(END_TRAIN))
    # MODEL = "../model/{}/".format(KEY_COLUMN + str(VER))
    os.makedirs(OUTPUT, exist_ok=True)
    # os.makedirs(MODEL, exist_ok=True)

    #key ids list
    KEY_IDS = list(pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')[KEY_COLUMN].unique())
    # KEY_IDS = list(pd.read_csv(ORIGINAL + 'sales_train_evaluation.csv')[KEY_COLUMN].unique())
    print("key id: {}\n".format(KEY_IDS))
    
    # Train Models
    params = {'boosting_type': 'gbdt',
              'objective': 'tweedie',
              'tweedie_variance_power': 1.1,
              'metric': 'rmse',
              'subsample': 0.5,
              'subsample_freq': 1,
              'seed': SEED,
              'learning_rate': 0.03,
              'num_leaves': 2**11-1,
              'min_data_in_leaf': 2**12-1,
              'feature_fraction': 0.5,
              'max_bin': 100,
              'boost_from_average': False,
              'verbose': -1,
              'num_threads':NUM_CPU} 


    # training every stores
    for key_id in KEY_IDS:
        print('Train', key_id)
        send_slack_notification("Train:{}".format(key_id))
        
        # Get grid for current store
        grid_df, features = get_data_by_key_column(KEY_COLUMN, TARGET, START_TRAIN, key_id)
        
        # mask
        train_mask = grid_df['d']<=END_TRAIN                                         # 0~1913 最終的には0~1941
        valid_mask = (END_TRAIN<grid_df['d']) & (grid_df['d']<=END_TRAIN+P_HORIZON)  # 1914~1941  予測対象 最終的には1942-1969
        preds_mask = grid_df['d']>(END_TRAIN-100)                                    # 1814~1969  再帰的予測のため100日分のbafferを取っている
         
        # Apply masks
        print("validのtargetにあるラベル数:", grid_df[valid_mask][TARGET].notnull().sum())
        train_data = lgb.Dataset(grid_df[train_mask][features], label=grid_df[train_mask][TARGET])
        valid_data = lgb.Dataset(grid_df[valid_mask][features], label=grid_df[valid_mask][TARGET])
        
        # save lgb dataset as bin to reduce memory
        train_data.save_binary('train_data.bin')
        train_data = lgb.Dataset('train_data.bin')
        
        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively = include '_tmp_'
        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        grid_df.to_pickle(OUTPUT + 'test_'+key_id+'.pkl')
        del grid_df

        # train
        model = lgb.train(params, train_data, valid_sets = [train_data,valid_data], num_boost_round=1400, verbose_eval = 100)

        # save the estimator as .bin
        model_name = 'lgb_model_'+key_id+'_v'+str(VER)+'.bin'  # 保存場所
        pickle.dump(model, open(OUTPUT + model_name, 'wb'))

        # Remove temporary files and objects 
        os.remove('train_data.bin')
        del train_data, valid_data, model
        gc.collect()

    t2 = time.time()
    send_slack_notification("FINISH")
    send_slack_notification("spend time: {}[min]".format(str((t2 - t1) / 60)))


if __name__ == "__main__":
    try:
        for KEY_COLUMN in ['store_id', 'dept_id', 'dept_store_id']:
            main(KEY_COLUMN)
    except:
        send_slack_error_notification("[ERROR]\n" + traceback.format_exc())
        print(traceback.format_exc())
