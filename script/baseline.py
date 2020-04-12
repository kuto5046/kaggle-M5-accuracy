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
"""
TODO validデータのおいて現在の予測と正解が最も乖離しているアイテムは何かをEDA
"""

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
        valid_preds = valid_preds.reshape(self.valid_target_columns.shape[0], self.valid_target_columns.shape[1])
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

    def feval(self, preds, dtrain):
        score = self.score(preds)
        return 'WRMSSE', score, False


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
    del train_df, calendar_df, sell_prices_df
    gc.collect()

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


def train_val_split(train_df):
    print("train/valid split ->")

    # train data
    train_day = train_df["d"].unique()[:-28]
    X_train = train_df[train_df["d"].isin(train_day)]
    y_train = X_train['demand']
    print("train(day): {0} ~ {1} -> {2}".format(train_day[0], train_day[-1], len(train_day)))

    # valid data
    valid_day = train_df["d"].unique()[-28:]
    X_val = train_df[train_df["d"].isin(valid_day)]
    y_val = X_val['demand']
    print("valid(day): {0} ~ {1} -> {2}".format(valid_day[0], valid_day[-1], len(valid_day)))

    del train_df
    gc.collect()

    return X_train, y_train, X_val, y_val


# def evaluate_WRMSSE(val_pred):
#     train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
#     calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
#     sell_prices_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

#     train_df = train_df.sort_values("id").reset_index(drop=True)
#     train_fold_df = train_df.iloc[:, :-28]
#     valid_fold_df = train_df.iloc[:, -28:]
#     del train_df
#     gc.collect()

#     evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar_df, sell_prices_df)
    
#     val_pred = val_pred.reshape(valid_fold_df.shape[0], valid_fold_df.shape[1])
#     columns = valid_fold_df.columns
#     val_pred = pd.DataFrame(val_pred, columns=columns)

#     return evaluator.score(val_pred)


def train_lgb(X_train, y_train, X_val, y_val, features, evaluator, date):
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
    train_set = lgb.Dataset(X_train[features], y_train)
    val_set = lgb.Dataset(X_val[features], y_val)

    # train/validation
    print("\n[START] training model ->")
    model = lgb.train(params, train_set, num_boost_round=1200, valid_sets=val_set, feval=evaluator.feval, verbose_eval=100)

    # save model
    model_dir = "../model/{}/".format(date[:8])
    os.makedirs(model_dir, exist_ok=True)
    model_name = "model_{}.pickle".format(date[9:])
    model_path = model_dir + model_name
    with open(model_path, mode='wb') as fp:
        pickle.dump(model, fp)

    val_pred = model.predict(X_val[features])
    val_RMSE_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(f'val RMSE score: {val_RMSE_score}')
    val_WRMSSE_score = evaluator.score(val_pred)
    print(f'val WRMSSE score: {val_WRMSSE_score}')

    return model


def predict_and_submmision(model, features, date):
    alpha = 1.018
    fday = datetime(2016, 4, 25) 

    te = create_data(is_train=False, num_train_day=56)
    cols = [f"F{i+1}" for i in range(28)]

    for tdelta in tqdm(range(28)):
        day = fday + timedelta(days=tdelta)
        tst = te[(te["date"] >= day - timedelta(days=56)) & (te["date"] <= day)].copy()
        tst = feature_engineering(tst)
        tst = tst.loc[tst["date"] == day, features]
        te.loc[te.date == day, "demand"] = alpha * model.predict(tst) # magic multiplier by kyakovlev
    
    stage1_sub = te.loc[te["d"] > "d_1913", ["id", "demand"]].copy()
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
    # pretrained_model = "../model/20200411/model_235756.pickle"

    t1 = time.time()
    date = datetime.today().strftime("%Y%m%d_%H%M%S")

    data_df = create_data(is_train=True, num_train_day =1500)
    data_df = feature_engineering(data_df)

    # define list of features
    default_features = ['item_id', 'dept_id', 'store_id','cat_id', 'state_id', 'event_name_1', 'event_type_1',
                        'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']
    demand_features = ['lag7', 'lag28', 'rmean_lag7_7', 'rmean_lag7_28', 'rmean_lag28_7', 'rmean_lag28_28']
    # price_features = ['sell_price_lag1', 'sell_price_lag7', 'sell_price_lag28', "diff_sell_price", "div_sell_price"]
    time_features = ["year", "month", "week", "quarter", "mday", "wday"]  # , "black_friday"]
    features = default_features + demand_features + time_features  # + price_features 
    print("N_features: {}\n".format(len(features)))

    # train/val split
    X_train, y_train, X_val, y_val = train_val_split(data_df)
    
    # evaluator
    train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
    calendar_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
    sell_prices_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
    train_df = train_df.sort_values("id").reset_index(drop=True)
    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:]
    del train_df
    gc.collect()

    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar_df, sell_prices_df)

    # train
    model = train_lgb(X_train, y_train, X_val, y_val, features, evaluator, date)
    # with open(pretrained_model, mode='rb') as fp:
    #     model = pickle.load(fp)

    # predict & submission
    predict_and_submmision(model, features, date)

    t2 = time.time()
    print("\nspend time: {}[min]".format(str((t2 - t1) / 60)))


if __name__ == "__main__":
    main()

