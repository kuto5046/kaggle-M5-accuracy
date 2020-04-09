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
from tqdm import tqdm
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')
"""
TODO 365 + 28日分をtrainデータとしてlag特徴量を作成し、古い28日分は欠損値が出るので削除

"""

# メモリ使用量の削減
def reduce_mem_usage(df, verbose=False):
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
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# function to read the data and merge it
# (ignoring some columns, this is a very fst model)
def read_data():
    print('Reading files...')

    calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar: ' + str(calendar_df.shape))

    sell_prices_df = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices: ' + str(sell_prices_df.shape))

    train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
    print('Sales train validation: ' + str(train_df.shape))

    submission_df = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
    print("Submission: " + str(submission_df.shape))

    return calendar_df, sell_prices_df, train_df, submission_df


def melt_and_merge(calendar_df, sell_prices_df, train_df, submission_df, num_train_data):

    # trainは直近１年間のデータのみ使用
    drop_columns = [f"d_{d}" for d in range(1, (1913 - num_train_data) + 1)]
    train_df.drop(drop_columns, inplace = True, axis=1)
    print("\ntrainは直近１年間のデータのみ使用")
    print('Sales train validation(remain only one year): ' + str(train_df.shape))

    # 商品情報を抽出
    product_df = train_df.loc[:, "id":"state_id"]

    # 列方向に連なっていたのを変形し行方向に連ねるように整理
    train_df = pd.melt(train_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                       var_name='day', value_name='demand')

    train_day = train_df["day"].unique()
    print("train_data: {0} ~ {1} -> {2}".format(train_day[0], train_day[-1], len(train_day)))

    # seperate test dataframes
    stage1_eval_df = submission_df[submission_df["id"].str.contains("validation")]
    stage2_eval_df = submission_df[submission_df["id"].str.contains("evaluation")]

    # change column names
    stage1_eval_df.columns = ["id"] + [f"d_{d}" for d in range(1914, 1942)]  # F1 ~ F28 => d_1914 ~ d_1941
    stage2_eval_df.columns = ["id"] + [f"d_{d}" for d in range(1942, 1970)]  # F1 ~ F28 => d_1942 ~ d_1969

    # melt, mergeを使ってsubmission用のdataframeを上のsales_train_validationと同様の形式に変形
    stage1_eval_df = stage1_eval_df.merge(product_df, how='left', on='id')
    stage1_eval_df = pd.melt(stage1_eval_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                             var_name='day', value_name='demand')
    stage1_day = stage1_eval_df["day"].unique()
    print("[STAGE1] eval_data: {0} ~ {1} -> {2}".format(stage1_day[0], stage1_day[-1], len(stage1_day)))

    # train_df, stage1_eval_dfと同様にstage2_eval_dfとproduct_dfをmergeさせたい
    # しかしidが_evaluationのままだとデータが一致せずmergeできないので一時的に_validationにidを変更
    stage2_eval_df['id'] = stage2_eval_df.loc[:, 'id'].str.replace('_evaluation', '_validation')
    stage2_eval_df = stage2_eval_df.merge(product_df, how='left', on='id')
    stage2_eval_df['id'] = stage2_eval_df.loc[:, 'id'].str.replace('_validation', '_evaluation')
    stage2_eval_df = pd.melt(stage2_eval_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                             var_name='day', value_name='demand')
    stage2_day = stage2_eval_df["day"].unique()
    print("[STAGE2] eval_data: {0} ~ {1} -> {2}".format(stage2_day[0], stage2_day[-1], len(stage2_day)))

    train_df['part'] = 'train'
    stage1_eval_df['part'] = 'stage1'
    stage2_eval_df['part'] = 'stage2'

    data_df = pd.concat([train_df, stage1_eval_df, stage2_eval_df], axis=0)
    data_df = reduce_mem_usage(data_df)

    # 不要なdataframeの削除
    del train_df, stage1_eval_df, stage2_eval_df, product_df

    # drop some calendar features
    calendar_df.drop(['weekday', 'wday', 'month', 'year'], inplace=True, axis=1)

    # delete stage2_eval_df for now
    data_df = data_df[data_df['part'] != 'stage2']
    print("[CHECK] Remove the stage2 eval data")

    # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
    data_df = pd.merge(data_df, calendar_df, how='left', left_on=['day'], right_on=['d'])
    data_df.drop('d', inplace=True, axis=1)

    # get the sell price data (this feature should be very important)
    data_df = data_df.merge(sell_prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    return data_df


# label encoding
def encode_categorical(data_df):
    nan_features = ['event_name_1', 'event_type_1',
                    'event_name_2', 'event_type_2']
    for feature in nan_features:
        # label encodingのためnanを文字列に変換
        data_df[feature].fillna('unknown', inplace=True)

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
           'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data_df[feature] = encoder.fit_transform(data_df[feature])

    return data_df


def data_split(data_df):
    # train data
    train_df = data_df[data_df['part'] == 'train']

    # stage1 eval data(特徴量生成用に56日分多めに確保しておく)
    test_df = data_df[(data_df['day'] > 'd_1857')]  # 56日前~(lag特徴量生成に使用) -> 1914(予測対象) ~

    del data_df
    gc.collect()

    return train_df, test_df


def feature_engineering(data_df):
    """
    1日後のリード特徴量
    1日前のラグ特徴量
    """

    # ラグ特徴量
    data_df['lag1'] = data_df.groupby(['id'])['demand'].shift(1)
    data_df['lag7'] = data_df.groupby(['id'])['demand'].shift(7)
    data_df['lag28'] = data_df.groupby(['id'])['demand'].shift(28)
    data_df['rmean_lag7_7'] = data_df.groupby(['id'])['lag7'].transform(lambda x: x.rolling(7).mean())
    data_df['rmean_lag7_28'] = data_df.groupby(['id'])['lag7'].transform(lambda x: x.rolling(28).mean())
    data_df['rmean_lag28_7'] = data_df.groupby(['id'])['lag28'].transform(lambda x: x.rolling(7).mean())
    data_df['rmean_lag28_28'] = data_df.groupby(['id'])['lag28'].transform(lambda x: x.rolling(28).mean())

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
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df['year'] = data_df['date'].dt.year.astype(np.int16)
    data_df['quarter'] = data_df['date'].dt.quarter.astype(np.int8)
    data_df['month'] = data_df['date'].dt.month.astype(np.int8)
    data_df['week'] = data_df['date'].dt.week.astype(np.int8)
    data_df['mday'] = data_df['date'].dt.day.astype(np.int8)
    data_df['wday'] = data_df['date'].dt.dayofweek.astype(np.int8)
    # data_df['is_year_end'] = data_df['date'].dt.is_year_end.astype(np.int8)
    # data_df['is_year_start'] = data_df['date'].dt.is_year_start.astype.astype(np.int8)
    # data_df['is_quarter_end'] = data_df['date'].dt.is_quarter_end.astype(np.int8)
    # data_df['is_quarter_start'] = data_df['date'].is_quarter_start.astype(np.int8)
    # data_df['is_month_end'] = data_df['date'].dt.is_month_end.astype(np.int8)
    # data_df['is_month_start'] = data_df['date'].dt.is_month_start.astype(np.int8)
    # data_df["is_weekend"] = data_df["dayofweek"].isin([5, 6]).astype(np.int8)

    # black friday
    # black_friday = ["2011-11-25", "2012-11-23", "2013-11-29", "2014-11-28", "2015-11-27"]
    # data_df["black_friday"] = data_df["date"].isin(black_friday) * 1
    # data_df["black_friday"] = data_df["black_friday"].astype(np.int8)

    # lag特徴量によって欠損している部分を削除
    data_df.dropna(inplace = True)
    data_df = reduce_mem_usage(data_df)

    return data_df


# 交差検証はランダムサンプリングで疑似的に行う
def train_val_split(train_df):

    # train data
    X_train = train_df[train_df['part'] <= 'train']
    y_train = X_train['demand']

    # valid data
    # This is just a subsample of the training set, not a real validation set !
    print("\n[CHECK] This valid data is not a real valid data, just random sampling data from train data!")
    fake_valid_inds = np.random.choice(len(X_train), 1000000)
    print(fake_valid_inds)
    X_val = X_train.iloc[fake_valid_inds]
    y_val = X_val["demand"]

    del train_df
    gc.collect()

    return X_train, y_train, X_val, y_val


# # 交差検証は時系列に沿ったhold-out法で行う
# def data_split(data_df):
#     # train data
#     X_train = data_df[data_df['day'] <= 'd_1885']
#     y_train = X_train['demand']

#     # valid data
#     X_val = data_df[(data_df['day'] > 'd_1885') & (data_df['day'] <= 'd_1913')]
#     y_val = X_val['demand']

#     # stage1 eval data
#     test = data_df[(data_df['day'] > 'd_1913')]

#     del data_df
#     gc.collect()

#     return X_train, y_train, X_val, y_val, test


def train_lgb(X_train, y_train, X_val, y_val, features, date):
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
              'bagging_freq': 1}

    # datasetの作成
    train_set = lgb.Dataset(X_train[features], y_train)
    val_set = lgb.Dataset(X_val[features], y_val)

    # train/validation
    print("\n[START] training model ->")
    model = lgb.train(params, train_set, num_boost_round=1500,  # early_stopping_rounds=200,
                      valid_sets=val_set, verbose_eval=100)

    # save model
    model_dir = "../model/{}/".format(date[:8])
    os.makedirs(model_dir, exist_ok=True)
    model_name = "model_{}.pickle".format(date[9:])
    model_path = model_dir + model_name
    with open(model_path, mode='wb') as fp:
        pickle.dump(model, fp)

    val_pred = model.predict(X_val[features], num_iteration=model.best_iteration)
    val_RMSE_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(f'val RMSE score: {val_RMSE_score}')

    return model


# min_lag日ずつ予測していく
def predict_lgb(model, test_df, features, min_lag):
    train_day = test_df[test_df["part"]=="train"]["day"].unique()  # 特徴量作成のための部分
    stage1_day = test_df[test_df["part"]=="stage1"]["day"].unique()  # 実際に予測する部分
    print("[START] predict ->")
    num_trials = int(len(stage1_day) / min_lag)  # 試行回数
    alpha = 1.02

    for i in tqdm(range(num_trials)):
        target_day = np.append(train_day, stage1_day[:(i + 1) * min_lag])  # 対象の指定(min_lag日ずつ対象範囲を広げていく)
        partial_test = test_df[test_df["day"].isin(target_day)].copy()  # 該当するデータを抽出
        partial_test = feature_engineering(partial_test)
        pred_target_day = stage1_day[i * min_lag:(i + 1) * min_lag]
    
        partial_test = partial_test[partial_test["day"].isin(pred_target_day)]  # 予測対象のみ抜粋(1日)
        partial_pred = model.predict(partial_test[features])

        test_df.loc[test_df["day"].isin(pred_target_day), "demand"] = alpha * partial_pred

    return test_df[test_df["part"]=="stage1"]


def result_submission(test_df, submission_df, date):
    output_dir = "../output/{}/".format(date[:8])
    os.makedirs(output_dir, exist_ok=True)
    submission_name = "submission_{}.csv".format(date[9:])
    submission_path = output_dir + submission_name
    
    pred = test_df[['id', 'date', 'demand']]
    pred = pd.pivot(pred, index='id', columns='date', values='demand').reset_index()
    pred.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    eval_rows = [row for row in submission_df['id'] if 'evaluation' in row]
    eval = submission_df[submission_df['id'].isin(eval_rows)]

    valid = submission_df[['id']].merge(pred, on='id')
    final = pd.concat([valid, eval])
    final.to_csv(submission_path, index=False)
    print("\ncompleted process.")


def main():
    # 変更パラメータ
    num_train_data = 1114  # 421 = 365 + 28*2
    min_lag = 1
    is_train = True  # Trueならモデルを訓練 Falseなら既存モデルを利用
    pretrained_model = "../model/20200408/model_184240.pickle"

    # read data
    t1 = time.time()
    date = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    calendar_df, sell_prices_df, train_df, submission_df = read_data()

    # preprocessing
    data_df = melt_and_merge(calendar_df, sell_prices_df, train_df, submission_df, num_train_data)
    data_df = encode_categorical(data_df)

    # data split
    train_df, test_df = data_split(data_df)

    # feature engineering
    if is_train:
        print("\n[START] feature engineering ->")
        train_df = feature_engineering(train_df)
        print("[FINISH] feature engineering")

        # train/val split
        X_train, y_train, X_val, y_val = train_val_split(train_df)

    # define list of features
    default_features = ['item_id', 'dept_id', 'store_id','cat_id', 'state_id', 'event_name_1', 'event_type_1',
                        'event_name_2', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']
    demand_features = ['lag1', 'lag7', 'lag28', 'rmean_lag7_7', 'rmean_lag7_28', 'rmean_lag28_7', 'rmean_lag28_28']
    # price_features = ['sell_price_lag1', 'sell_price_lag7', 'sell_price_lag28', "diff_sell_price", "div_sell_price"]
    time_features = ["year", "month", "week", "quarter", "mday", "wday"]  # , "black_friday"]
    features = default_features + demand_features + time_features  # + price_features 
    print("N_features: {}\n".format(len(features)))

    # train
    if is_train:
        model = train_lgb(X_train, y_train, X_val, y_val, features, date)
    else:
        with open(pretrained_model, mode='rb') as fp:
            model = pickle.load(fp)

    # predict
    test_df = predict_lgb(model, test_df, features, min_lag)

    # submission
    result_submission(test_df, submission_df, date)

    t2 = time.time()
    print("\nspend time: {}[min]".format(str((t2 - t1) / 60)))


if __name__ == "__main__":
    main()

