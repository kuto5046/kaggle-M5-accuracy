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


class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)


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


# Read data every stores
def get_data_by_store(KEY_COLUMN, TARGET, START_TRAIN, key_id):

    #PATHS for Features
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
    
    df["d"] = df["d"].str.strip("d_").astype(int)

    # Create featur es list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET] + features]
    
    # Skipping first n rows
    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    
    return df, features


# Recombine Test set after training
# def get_base_test(KEY_COLUMN, KEY_IDS, OUTPUT):
#     base_test = pd.DataFrame()

#     for key_id in KEY_IDS:
#         temp_df = pd.read_pickle(OUTPUT + 'test_'+key_id+'.pkl')
#         temp_df[KEY_COLUMN] = key_id
#         base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
#     return base_test


def main():
    t1 = time.time()

    # var
    VER = 1                          # Our model version
    TARGET = "sales"
    KEY_COLUMN = 'dept_id'          # training each id
    NUM_CPU = 8
    SEED = 5046                      # We want all things
    seed_everything(SEED)            # to be as deterministic 

    NOW_DATE = datetime.today().strftime("%Y%m%d_%H%M%S")
    ORIGINAL = '../input/m5-forecasting-accuracy/' 
    OUTPUT = '../output/{}/'.format(NOW_DATE[4:13])
    MODEL = "../model/{}/".format(NOW_DATE[4:13])  
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(MODEL, exist_ok=True)

    #LIMITS and const
    START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
    END_TRAIN   = 1913               # End day of our train set
    P_HORIZON   = 28                 # Prediction horizon

    #key ids list
    KEY_IDS = list(pd.read_csv(ORIGINAL + 'sales_train_validation.csv')[KEY_COLUMN].unique())
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
              'n_estimators': 1400,
              'boost_from_average': False,
              'verbose': -1,
              'num_threads':NUM_CPU} 

    # training every stores
    for key_id in KEY_IDS:
        print('Train', key_id)
        
        # Get grid for current store
        grid_df, features = get_data_by_store(KEY_COLUMN, TARGET, START_TRAIN, key_id)
        
        # mask
        train_mask = grid_df['d']<=END_TRAIN                            # 0~1913  TODO 訓練に予測対象が入っているのはいいの？
        valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))    # 1886~1913  予測対象
        preds_mask = grid_df['d']>(END_TRAIN-100)                       # 1813~1941  再帰的予測のため100日分のbafferを取っている
        
        # Apply masks
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

        # Launch seeder again to make lgb training 100% deterministic
        # seed_everything(SEED)

        # train
        model = lgb.train(params, train_data, valid_sets = [valid_data], verbose_eval = 100)
        y_pred = model.predict(grid_df[valid_mask][features])
        
        # train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
        # train_fold_df = train_df.iloc[:, :-28]
        # valid_fold_df = train_df.iloc[:, -28:]
        # valid_preds = valid_fold_df.copy() + np.random.randint(100, size=valid_fold_df.shape)

        # evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
        # evaluator.score(valid_preds)

        # save the estimator as .bin
        model_name = 'lgb_model_'+key_id+'_v'+str(VER)+'.bin'  # 保存場所
        pickle.dump(model, open(MODEL + model_name, 'wb'))

        # Remove temporary files and objects 
        os.remove('train_data.bin')
        del train_data, valid_data, model
        gc.collect()

    t2 = time.time()
    send_slack_notification("FINISH")
    send_slack_notification("spend time: {}[min]".format(str((t2 - t1) / 60)))


if __name__ == "__main__":
    try:
        main()
    except:
        send_slack_error_notification("[ERROR]\n" + traceback.format_exc())
        print(traceback.format_exc())
