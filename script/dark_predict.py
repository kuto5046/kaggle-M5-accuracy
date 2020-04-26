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

warnings.filterwarnings('ignore')

def send_slack_notification(message):
    webhook_url = 'https://hooks.slack.com/services/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'  # 終わったら無効化する
    data = json.dumps({'text': message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)


def send_slack_error_notification(message):
    webhook_url = 'https://hooks.slack.com/services/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'  # 終わったら無効化する
    data = json.dumps({"text":":no_entry_sign:" + message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    

## Multiprocess Runs
def df_parallelize_run(func, t_split):
    N_CORES = psutil.cpu_count()     # Available CPU cores
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


# Read data every stores
def get_features(TARGET):
    BASE     = '../input/m5-simple-fe/grid_part_1.pkl'
    PRICE    = '../input/m5-simple-fe/grid_part_2.pkl'
    CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
    LAGS     = '../input/m5-lags-features/lags_df_28.pkl'
    MEAN_ENC = '../input/m5-custom-features/mean_encoding_df.pkl'

    # FEATURES to remove
    # These features lead to overfit or values not present in test set
    remove_features = ['id','state_id','store_id', 'date','wm_yr_wk','d', TARGET]
    mean_features  = ['enc_cat_id_mean','enc_cat_id_std',
                    'enc_dept_id_mean','enc_dept_id_std',
                    'enc_item_id_mean','enc_item_id_std'] 

    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)

    # Leave only relevant store
    df = df[df['store_id']=="CA_1"]

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
    
    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    del df
    
    return features


# Recombine Test set after training
def get_base_test(STORES_IDS, OUTPUT):
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle(OUTPUT + 'test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    return base_test


# Helper to make dynamic rolling lags
# def make_lag(LAG_DAY):
#     lag_df = base_test[['id','d',TARGET]]
#     col_name = 'sales_lag_'+str(LAG_DAY)
#     lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
#     return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    TARGET = "sales"
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


def submission(all_preds, ORIGINAL, VER):
    """
    Reading competition sample submission and
    merging our predictions
    As we have predictions only for "_validation" data
    we need to do fillna() for "_evaluation" items
    """
    submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv('submission_v'+str(VER)+'.csv', index=False)


try:
    t1 = time.time()

    # var
    VER = 1                          # Our model version
    TARGET = "sales"
    SEED = 5046                      # We want all things
    seed_everything(SEED)            # to be as deterministic 
    TRAINED_DIR = "20200425/"
    #LIMITS and const
    END_TRAIN   = 1913               # End day of our train set

    #PATHS for Features
    ORIGINAL = '../input/m5-forecasting-accuracy/'
    MODEL = "../model/" + TRAINED_DIR
    OUTPUT = '../output/' + TRAINED_DIR

    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(MODEL, exist_ok=True)

    # SPLITS for lags creation
    # SHIFT_DAY  = 28
    # N_LAGS     = 15
    # LAGS_SPLIT = [col for col in range(SHIFT_DAY, SHIFT_DAY + N_LAGS)]
    ROLS_SPLIT = []
    for i in [1,7,14]:
        for j in [7,14,30,60]:
            ROLS_SPLIT.append([i,j])


    features = get_features(TARGET)

    # Create Dummy DataFrame to store predictions
    all_preds = pd.DataFrame()

    # Join back the Test dataset with 
    # a small part of the training data 
    # to make recursive features
    STORES_IDS = list(pd.read_csv(ORIGINAL + 'sales_train_validation.csv')['store_id'].unique())
    base_test = get_base_test(STORES_IDS, OUTPUT)

    # Timer to measure predictions time 
    main_time = time.time()

    # Loop over each prediction day
    # As rolling lags are the most timeconsuming
    # we will calculate it for whole day
    for PREDICT_DAY in range(1,29):    
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()

        # Make temporary grid to calculate rolling lags
        grid_df = base_test.copy()
        grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)  # TODO 挙動理解
        
        for store_id in STORES_IDS:
            
            # Read all our models and make predictions
            # for each day/store pairs
            model_name = 'lgb_model_'+store_id+'_v'+str(VER)+'.bin' 

            # if USE_AUX:
            #     model_name = AUX_MODELS + model_name
            
            model = pickle.load(open(MODEL + model_name, 'rb'))
            
            day_mask = base_test['d']==(END_TRAIN + PREDICT_DAY)
            store_mask = base_test['store_id'] == store_id
            
            mask = (day_mask)&(store_mask)
            base_test[TARGET][mask] = model.predict(grid_df[mask][features])
        
        # Make good column naming and add to all_preds DataFrame
        temp_df = base_test[day_mask][['id',TARGET]]
        temp_df.columns = ['id','F'+str(PREDICT_DAY)]
        if 'id' in list(all_preds):
            all_preds = all_preds.merge(temp_df, on=['id'], how='left')
        else:
            all_preds = temp_df.copy()
            
        print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                        ' %0.2f min total |' % ((time.time() - main_time) / 60),
                        ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
        del temp_df
        
    all_preds = all_preds.reset_index(drop=True)

    submission(all_preds, ORIGINAL, VER)

    t2 = time.time()
    send_slack_notification("FINISH")
    send_slack_notification("spend time: {}[min]".format(str((t2 - t1) / 60)))

except:
    send_slack_error_notification("[ERROR]\n" + traceback.format_exc())
    print(traceback.format_exc())
