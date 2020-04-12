import gc
import os
import time
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from typing import Union
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

# function to read the data and merge it
# TODO 中身を一度確認しつつコードをより分かりやすく整理する
def create_data(is_train, num_train_day):
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
                "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
                "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

    print("\n[STSRT] read data ->")
    sell_prices_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            sell_prices_df[col] = sell_prices_df[col].cat.codes.astype("int16")
            sell_prices_df[col] -= sell_prices_df[col].min()
            
    calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            calendar_df[col] = calendar_df[col].cat.codes.astype("int16")
            calendar_df[col] -= calendar_df[col].min()
    
    numcols = [f"d_{day}" for day in range(1914 - num_train_day, 1914)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    train_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 
                            usecols = catcols + numcols, dtype = dtype)

    for col in catcols:
        if col != "id":
            train_df[col] = train_df[col].cat.codes.astype("int16")
            train_df[col] -= train_df[col].min()
    
    if not is_train:
        for day in range(28):
            train_df[f"d_{day+1914}"] = np.nan
    
    data_df = pd.melt(train_df, id_vars = catcols, value_vars = [col for col in train_df.columns if col.startswith("d_")], 
                      var_name = "d", value_name = "demand")
    data_df = data_df.merge(calendar_df, on= "d", copy = False)
    data_df = data_df.merge(sell_prices_df, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    all_day = data_df["d"].unique()
    print("data: {0} ~ {1} -> {2}".format(all_day[0], all_day[-1], len(all_day)))
    
    return data_df


def feature_engineering(data_df):
    # lag features
    # data_df['lag1'] = data_df[["id", "demand"]].groupby('id')['demand'].shift(1)
    data_df['lag7'] = data_df[["id", "demand"]].groupby('id')['demand'].shift(7)
    data_df['lag28'] = data_df[["id", "demand"]].groupby('id')['demand'].shift(28)

    # data_df["rmean_lag1_7"] = data_df[["id", "lag1"]].groupby("id")["lag1"].transform(lambda x : x.rolling(7).mean())
    # data_df["rmean_lag1_28"] = data_df[["id", "lag1"]].groupby("id")["lag1"].transform(lambda x : x.rolling(28).mean())
    data_df["rmean_lag7_7"] = data_df[["id", "lag7"]].groupby("id")["lag7"].transform(lambda x : x.rolling(7).mean())
    data_df["rmean_lag7_28"] = data_df[["id", "lag7"]].groupby("id")["lag7"].transform(lambda x : x.rolling(28).mean())
    data_df["rmean_lag28_7"] = data_df[["id", "lag28"]].groupby("id")["lag28"].transform(lambda x : x.rolling(7).mean())
    data_df["rmean_lag28_28"] = data_df[["id", "lag28"]].groupby("id")["lag28"].transform(lambda x : x.rolling(28).mean())

    # price features
    # data_df['sell_price_lag1'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    # data_df['sell_price_lag7'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(7))
    # data_df['sell_price_lag28'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(28))
    # mean_sell_price_df = data_df.groupby('id').mean()
    # mean_sell_price_df.rename(columns={"sell_price": "mean_sell_price"}, inplace=True)
    # data_df = data_df.merge(mean_sell_price_df["mean_sell_price"], on="id")
    # data_df["diff_sell_price"] = data_df["sell_price"] - data_df["mean_sell_price"]
    # data_df["div_sell_price"] = data_df["sell_price"] / data_df["mean_sell_price"]

    # time features
    data_df['year'] = data_df['date'].dt.year.astype(np.int16)
    data_df['quarter'] = data_df['date'].dt.quarter.astype(np.int8)
    data_df['month'] = data_df['date'].dt.month.astype(np.int8)
    data_df['week'] = data_df['date'].dt.week.astype(np.int8)
    data_df['mday'] = data_df['date'].dt.day.astype(np.int8)
    data_df['wday'] = data_df['date'].dt.dayofweek.astype(np.int8)

    # black_friday = ["2011-11-25", "2012-11-23", "2013-11-29", "2014-11-28", "2015-11-27"]
    # data_df["black_friday"] = data_df["date"].isin(black_friday) * 1
    # data_df["black_friday"] = data_df["black_friday"].astype(np.int8)

    return data_df


# 交差検証はランダムサンプリングで疑似的に行う
# TODO validationの工夫
def train_val_split(train_df, features):

    # train data
    X_train = train_df[features]
    y_train = train_df['demand']

    # valid data
    # This is just a subsample of the training set, not a real validation set !
    print("\n[CHECK] This valid data is not a real valid data, just random sampling data from train data!")

    fake_valid_inds = np.random.choice(len(X_train), 2000000, replace=False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)

    X_train = X_train.loc[train_inds]
    X_train = y_train.loc[train_inds]
    X_val = X_train.iloc[fake_valid_inds]
    y_val = y_train.iloc[fake_valid_inds]

    print("train:{}".format(len(X_train)/(len(X_train)+len(X_val))))
    print("valid:{}".format(len(X_val)/(len(X_train)+len(X_val))))

    del train_df
    gc.collect()

    return X_train, y_train, X_val, y_val




def train_lgb(X_train, y_train, X_val, y_val, date):
    # define random hyperparammeters
    params = {'boosting_type': 'gbdt',
              'metric': 'rmse',
              'objective': "poisson",
              'n_jobs': -1,
              'seed': 5046,
              'learning_rate': 0.075,
              'lambda_l2': 0.1,
              'sub_feature': 0.8,
              'sub_row': 0.75,
              'bagging_freq': 1,
              'num_leaves': 128,
              'min_data_in_leaf': 100
              }

    # datasetの作成
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val)

    # train/validation
    print("\n[START] training model ->")
    model = lgb.train(params, train_set, num_boost_round=1200, valid_sets=val_set, verbose_eval=100)

    # save model
    model_dir = "../model/{}/".format(date[:8])
    os.makedirs(model_dir, exist_ok=True)
    model_name = "model_{}.pickle".format(date[9:])
    model_path = model_dir + model_name
    with open(model_path, mode='wb') as fp:
        pickle.dump(model, fp)

    val_pred = model.predict(X_val)
    val_RMSE_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(f'val RMSE score: {val_RMSE_score}')

    return model


def predict_and_submmision(model, features, date):
    alpha = 1.018
    fday = datetime(2016, 4, 25) 

    te = create_data(is_train=False, num_train_day=56)
    cols = [f"F{i+1}" for i in range(28)]

    for tdelta in tqdm(range(0, 28)):
        day = fday + timedelta(days=tdelta)
        tst = te[(te["date"] >= day - timedelta(days=56)) & (te["date"] <= day)].copy()
        tst = feature_engineering(tst)
        tst = tst.loc[tst.date == day , features]
        te.loc[te.date == day, "demand"] = alpha * model.predict(tst) # magic multiplier by kyakovlev

    stage1_sub = te.loc[te["date"] >= fday, ["id", "demand"]].copy()
    stage1_sub["F"] = [f"F{rank}" for rank in stage1_sub.groupby("id")["id"].cumcount()+1]
    stage1_sub = stage1_sub.set_index(["id", "F" ]).unstack()["demand"][cols].reset_index()
    stage1_sub.fillna(0., inplace = True)
    stage1_sub.sort_values("id", inplace = True)
    stage1_sub.reset_index(drop=True, inplace = True)

    stage2_sub = stage1_sub.copy()
    stage2_sub["id"] = stage2_sub["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([stage1_sub, stage2_sub], axis=0, sort=False)

    output_dir = "../output/{}/".format(date[:8])
    os.makedirs(output_dir, exist_ok=True)
    submission_name = "submission_{}.csv".format(date[9:])
    submission_path = output_dir + submission_name
    sub.to_csv(submission_path, index=False)


def main():
    # 変更パラメータ
    pretrained_model = "../model/20200411/model_235756.pickle"

    t1 = time.time()
    date = datetime.today().strftime("%Y%m%d_%H%M%S")

    # data_df = create_data(is_train=True, num_train_day=1500)
    # data_df = feature_engineering(data_df)

    # define list of features
    default_features = ['item_id', 'dept_id', 'store_id','cat_id', 'state_id', 'event_name_1', 'event_type_1',
                        'event_name_2', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']
    demand_features = ['lag7', 'lag28', 'rmean_lag7_7', 'rmean_lag7_28', 'rmean_lag28_7', 'rmean_lag28_28']
    # price_features = ['sell_price_lag1', 'sell_price_lag7', 'sell_price_lag28', "diff_sell_price", "div_sell_price"]
    time_features = ["year", "month", "week", "quarter", "mday", "wday"]  # , "black_friday"]
    features = default_features + demand_features + time_features  # + price_features 
    print("N_features: {}\n".format(len(features)))

    # train/val split
    # X_train, y_train, X_val, y_val = train_val_split(data_df, features)
    
    # train
    # model = train_lgb(X_train, y_train, X_val, y_val, date)
    with open(pretrained_model, mode='rb') as fp:
        model = pickle.load(fp)

    # predict & submission
    predict_and_submmision(model, features, date)

    t2 = time.time()
    print("\nspend time: {}[min]".format(str((t2 - t1) / 60)))


if __name__ == "__main__":
    main()

