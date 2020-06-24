import gc
import json
import os
import pickle
import random
import time
from typing import Union
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
from evaluator import WRMSSEEvaluator
from utils import seed_everything, send_slack_notification, send_slack_error_notification

warnings.filterwarnings('ignore')

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
    if KEY_COLUMN == "dept_store_id":
        remove_features = ['id','state_id', KEY_COLUMN, 'dept_id', 'store_id', 'date','wm_yr_wk','d', TARGET]
    else:    
        remove_features = ['id','state_id', KEY_COLUMN, 'dept_store_id', 'date','wm_yr_wk','d', TARGET]

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
def get_base_test(KEY_COLUMN, KEY_IDS, OUTPUT):
    base_test = pd.DataFrame()

    for key_id in KEY_IDS:
        temp_df = pd.read_pickle(OUTPUT + 'test_'+key_id+'.pkl')
        temp_df[KEY_COLUMN] = key_id  # .astype("category")
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    return base_test


def make_lag_roll(LAG_DAY):
    TARGET = "sales"
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


def submission(all_preds, ORIGINAL, KEY_COLUMN, OUTPUT, VER, WRMSSEscore, END_TRAIN):
    """
    Reading competition sample submission and
    merging our predictions
    As we have predictions only for "_validation" data
    we need to do fillna() for "_evaluation" items
    """
    submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]

    if END_TRAIN == 1941:
        data = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
        data["id"] = data["id"].str.replace("evaluation", "validation")
        stage1 = data.iloc[:, -28:]
        stage1.insert(0, 'id', data['id'])
        stage1.columns =["id"] + ['F'+str(i) for i in range(1,29)]
        submission = submission.merge(stage1, on=['id'], how='left').fillna(0)
    else:
        all_preds["id"] = all_preds["id"].str.replace("evaluation", "validation")

    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv(OUTPUT + 'sub_v'+str(VER) + "_" + KEY_COLUMN + "_" + str(round(WRMSSEscore, 3)) + '.csv', index=False)


# main関数に該当
for KEY_COLUMN in ["store_id", "dept_id", "dept_store_id"]:
    try:
        t1 = time.time()

        # var
        VER = 1                          # Our model version
        TARGET = "sales"
        SEED = 5046                      # We want all things
        seed_everything(SEED)            # to be as deterministic 
        END_TRAIN   = 1941               # TODO  最後は1941に変更 End day of our train set
        P_HORIZON   = 28                 # Prediction horizon

        #PATHS for Features
        ORIGINAL = '../input/m5-forecasting-accuracy/'
        # TRAINED_DIR = "0612_0040/"
        # MODEL = "../model/" + TRAINED_DIR
        # OUTPUT = '../output/' + TRAINED_DIR
        OUTPUT = '../output/{}/'.format("v" + str(VER) + "_" + KEY_COLUMN + "_" + str(END_TRAIN))

        os.makedirs(OUTPUT, exist_ok=True)
        # os.makedirs(MODEL, exist_ok=True)

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
        KEY_IDS = list(pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')[KEY_COLUMN].unique())
        # KEY_IDS = list(pd.read_csv(ORIGINAL + 'sales_train_evaluation.csv')[KEY_COLUMN].unique())
        base_test = get_base_test(KEY_COLUMN, KEY_IDS, OUTPUT)

        # Timer to measure predictions time 
        main_time = time.time()

        # Loop over each prediction day
        # As rolling lags are the most timeconsuming
        # we will calculate it for whole day
        for PREDICT_DAY in range(1,29):    
            print('Predict | Day:', PREDICT_DAY)
            send_slack_notification('Predict | Day:{}'.format(PREDICT_DAY))
            start_time = time.time()

            # Make temporary grid to calculate rolling lags
            grid_df = base_test.copy()
            grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
            
            for key_id in KEY_IDS:
                
                # Read all our models and make predictions
                # for each day/store pairs
                model_name = 'lgb_model_'+key_id+'_v'+str(VER)+'.bin' 
                model = pickle.load(open(OUTPUT + model_name, 'rb'))
                
                day_mask = base_test['d']==(END_TRAIN + PREDICT_DAY)
                key_mask = base_test[KEY_COLUMN] == key_id
                
                mask = (day_mask)&(key_mask)
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
            del grid_df, temp_df
            
        all_preds = all_preds.reset_index(drop=True)
        evaluator = WRMSSEEvaluator()
        WRMSSEscore = evaluator.score(all_preds.iloc[:, 1:].to_numpy())
        print("WRMSSE: ", WRMSSEscore)
        
        submission(all_preds, ORIGINAL, KEY_COLUMN, OUTPUT, VER, WRMSSEscore, END_TRAIN)

        t2 = time.time()
        send_slack_notification("FINISH")
        send_slack_notification("spend time: {}[min]".format(str((t2 - t1) / 60)))

    except:
        send_slack_error_notification("[ERROR]\n" + traceback.format_exc())
        print(traceback.format_exc())
