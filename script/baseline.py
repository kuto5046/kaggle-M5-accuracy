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
NOTE
very fast model created by @ragnar

"""


class WRMSSEEvaluator(object):

    def __init__(self, data, product):
        self.data = data
        self.product = product
        self.NUM_ITEMS = 30490
        self.DAYS_PRED = 28
        self.train_start_day = 1549  # 変更する
        self.train_end_day = 1913


    def weight_calc(self):

        weight_mat = np.c_[np.ones([self.NUM_ITEMS, 1]).astype(np.int8),  # level 1
                           pd.get_dummies(self.product.state_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.store_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.cat_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.dept_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.state_id.astype(str) + self.product.cat_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.state_id.astype(str) + self.product.dept_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.store_id.astype(str) + self.product.cat_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.store_id.astype(str), drop_first=False).astype('int8').values,
                           pd.get_dummies(self.product.state_id.astype(str) + self.product.item_id.astype(str), drop_first=False).astype('int8').values,
                           np.identity(self.NUM_ITEMS).astype(np.int8)  # item :level 12
                           ].T
        weight_mat_csr = csr_matrix(weight_mat)
        
        del weight_mat
        gc.collect()

        # calculate the denominator of RMSSE, and calculate the weight base on sales amount
        train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
        d_name = ['d_' + str(i) for i in range(self.train_start_day, self.train_end_day+1)]
        train_df = weight_mat_csr * train_df[d_name].values

        # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
        # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
        df_tmp = ((train_df > 0) * np.tile(np.arange(self.train_start_day, self.train_end_day+1), (weight_mat_csr.shape[0], 1)))
        start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1) - 1
        flag = np.dot(np.diag(1/(start_no+1)), 
                      np.tile(np.arange(self.train_start_day, self.train_end_day+1), 
                      (weight_mat_csr.shape[0], 1))) < 1
    
        train_df = np.where(flag, np.nan, train_df)

        # denominator of RMSSE / RMSSEの分母
        weight1 = np.nansum(np.diff(train_df, axis=1) ** 2, axis=1)/(self.train_end_day - start_no)

        # calculate the sales amount for each item/level
        df_tmp = self.data[(self.data['day'] > 'd_1885') & (self.data['day'] <= 'd_1913')]
        df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
        df_tmp = df_tmp.groupby(['id'])['amount'].apply(np.sum)
        df_tmp = df_tmp[self.product.id].values

        weight2 = weight_mat_csr * df_tmp
        weight2 = weight2/np.sum(weight2)

        del train_df, df_tmp
        gc.collect()

        return weight1, weight2, weight_mat_csr

    def wrmsse(self, preds, data):

        weight1, weight2, weight_mat_csr = self.weight_calc()

        # this function is calculate for last 28 days to consider the non-zero demand period
        # actual obserbed values
        y_true = data.get_label()
        y_true = y_true[-(self.NUM_ITEMS * self.DAYS_PRED):]
        preds = preds[-(self.NUM_ITEMS * self.DAYS_PRED):]

        # number of columns
        num_col = self.DAYS_PRED

        # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
        preds = preds.reshape(num_col, self.NUM_ITEMS).T
        true = y_true.reshape(num_col, self.NUM_ITEMS).T

        train = weight_mat_csr * np.c_[preds, true]

        score = np.sum(np.sqrt(np.mean(np.square( train[:, :num_col] - train[:, num_col:]), axis=1) / weight1) * weight2)

        return 'wrmsse', score, False


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


def melt_and_merge(calendar_df, sell_prices_df, train_df, submission_df):

    # trainは直近１年間のデータのみ使用
    day_per_year = 365
    train_end_day = 1913
    drop_columns = [f"d_{d}" for d in range(1, (train_end_day - day_per_year) + 1)]
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
    valid_df = submission_df[submission_df["id"].str.contains("validation")]
    eval_df = submission_df[submission_df["id"].str.contains("evaluation")]

    # change column names
    # validation data: F1 ~ F28 => d_1914 ~ d_1941
    valid_df.columns = ["id"] + [f"d_{d}" for d in range(1914, 1942)]
    # evaluation data: F1 ~ F28 => d_1942 ~ d_1969
    eval_df.columns = ["id"] + [f"d_{d}" for d in range(1942, 1970)]

    # melt, mergeを使ってsubmission用のdataframeを上のsales_train_validationと同様の形式に変形
    valid_df = valid_df.merge(product_df, how='left', on='id')
    valid_df = pd.melt(valid_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                       var_name='day', value_name='demand')
    valid_day = valid_df["day"].unique()
    print("valid_data: {0} ~ {1} -> {2}".format(valid_day[0], valid_day[-1], len(valid_day)))

    # train_df, valid_dfと同様にeval_dfとproduct_dfをmergeさせたい
    # しかしidが_evaluationのままだとデータが一致せずmergeできないので一時的に_validationにidを変更
    eval_df['id'] = eval_df.loc[:, 'id'].str.replace('_evaluation', '_validation')
    eval_df = eval_df.merge(product_df, how='left', on='id')
    eval_df['id'] = eval_df.loc[:, 'id'].str.replace('_validation', '_evaluation')
    eval_df = pd.melt(eval_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                      var_name='day', value_name='demand')
    eval_day = eval_df["day"].unique()
    print("eval_data: {0} ~ {1} -> {2}".format(eval_day[0], eval_day[-1], len(eval_day)))

    train_df['part'] = 'train'
    valid_df['part'] = 'valid'
    eval_df['part'] = 'eval'

    data_df = pd.concat([train_df, valid_df, eval_df], axis=0)
    data_df = reduce_mem_usage(data_df)
    # print("\n[INFO] data_df(after merge valid & eval) ->")
    # data_df.head()

    # 不要なdataframeの削除
    del train_df, valid_df, eval_df

    # drop some calendar features
    calendar_df.drop(['weekday', 'wday', 'month', 'year'],
                     inplace=True, axis=1)

    # delete eval_df for now
    data_df = data_df[data_df['part'] != 'eval']
    print("[CHECK] Remove the eval data")

    # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
    data_df = pd.merge(data_df, calendar_df, how='left',
                       left_on=['day'], right_on=['d'])
    data_df.drop('d', inplace=True, axis=1)

    # get the sell price data (this feature should be very important)
    data_df = data_df.merge(
        sell_prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    # print("\n[INFO] data_df(after merge calendar & prices) ->")
    # print(data_df.head(5))
    # print(data_df.columns)

    return data_df, product_df


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


# 特徴量エンジニアリング
def feature_engineering(data_df):
    """
    1日後のリード特徴量
    1日前のラグ特徴量
    """

    print("\n[START] feature engineering ->")

    # black friday
    # black_friday = ["2011-11-25", "2012-11-23", "2013-11-29", "2014-11-28", "2015-11-27"]
    # data_df["black_friday"] = data_df["date"].isin(black_friday) * 1

    # rolling demand features
    # 28個分下にずらすことで時系列データの差分を取る
    # data_df['lag_7'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(7))
    data_df['lag_28'] = data_df.groupby(
        ['id'])['demand'].transform(lambda x: x.shift(28))

    # per a week
    # data_df['rmean_lag7_t7'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(7).rolling(7).mean())
    # data_df['rmean_lag7_t28'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(7).rolling(28).mean())
    data_df['rmean_lag28_t7'] = data_df.groupby(
        ['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data_df['rmean_lag28_t28'] = data_df.groupby(
        ['id'])['demand'].transform(lambda x: x.shift(28).rolling(28).mean())

    # data_df['rolling_std_t7'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    # per a month
    # data_df['rolling_mean_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    # data_df['rolling_mean_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(7).rolling(30).mean())
    # data_df['rolling_std_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())

    # per 2 month
    # data_df['rolling_mean_t60'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
    # data_df['rolling_std_t60'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).std())

    # per 3 month
    # data_df['rolling_mean_t90'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    # data_df['rolling_std_t90'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).std())

    # half year
    # data_df['rolling_mean_t180'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    # data_df['rolling_std_t180'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).std())

    # per a month
    # data_df['rolling_std_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    # data_df['rolling_skew_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
    # data_df['rolling_kurt_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())

    # price features
    # data_df['lag_price_t1'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    # data_df['price_change_t1'] = (data_df['lag_price_t1'] - data_df['sell_price']) / (data_df['lag_price_t1'])
    # data_df['rolling_price_max_t365'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    # data_df['price_change_t365'] = (data_df['rolling_price_max_t365'] - data_df['sell_price']) / (data_df['rolling_price_max_t365'])
    # data_df['rolling_price_std_t7'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    # data_df['rolling_price_std_t30'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    # data_df.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)

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

    # mean_sell_price_df = data_df.groupby('id').mean()
    # mean_sell_price_df.rename(columns={"sell_price": "mean_sell_price"}, inplace=True)
    # data_df = data_df.merge(mean_sell_price_df["mean_sell_price"], on="id")
    # data_df["diff_sell_price"] = data_df["mean_sell_price"] - data_df["sell_price"]
    print("[FINISH] feature engineering")

    return data_df


# 時系列データの時はold -> train/val/test -> new とする？
def data_split(data_df):
    # going to evaluate with the last 28 days
    X_train = data_df[data_df['day'] <= 'd_1885']
    y_train = X_train['demand']

    X_val = data_df[(data_df['day'] > 'd_1885') & (data_df['day'] <= 'd_1913')]
    y_val = X_val['demand']

    test = data_df[(data_df['day'] > 'd_1913')]

    del data_df
    gc.collect()

    return X_train, y_train, X_val, y_val, test


def run_lgb(X_train, y_train, X_val, y_val, test, date, evaluator):
    # define list of features
    # 学習/予測に使用する特徴量
    # default_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    #                     'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']
    # demand_features = ['lag_t28', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30',
    #                    'rolling_std_t30', 'rolling_mean_t60', 'rolling_std_t90', 'rolling_std_t180']
    # price_features = ["price_change_t1", "price_change_t365", "rolling_price_std_t7", "diff_sell_price"]
    # time_features = ["year", "month", "dayofweek", "is_month_end", "is_month_start", "is_weekend"]
    # add_features = ['black_friday']
    # features = default_features + demand_features + price_features + time_features + add_features

    default_features = ['item_id', 'dept_id', 'cat_id', 'state_id', 'event_name_1', 'snap_CA', 'snap_WI', 'sell_price']
    demand_features = ['lag_28', 'rmean_lag28_t7', 'rmean_lag28_t28']
    # demand_features = ['lag_28', 'lag_7', 'rmean_lag7_t7', 'rmean_lag7_t28', 'rmean_lag28_t7', 'rmean_lag28_t28']
    time_features = ["year", "month", "week", "mday", "wday"]
    features = default_features + demand_features + time_features
    print("N_features: {}\n".format(len(features)))

    # define random hyperparammeters
    params = {'boosting_type': 'gbdt',
              'metric': 'custom',
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
    model = lgb.train(params, train_set, num_boost_round=2, # early_stopping_rounds=200,
                      valid_sets=[train_set, val_set], verbose_eval=1, feval=evaluator.wrmsse)

    # save model
    model_dir = "../model/"
    model_name = "model_{}.pickle".format(date)
    model_path = model_dir + model_name
    os.makedirs(model_path, exist_ok=True)
    with open(model_path + 'model.pickle', mode='wb') as fp:
        pickle.dump(model, fp)

    # predict
    val_pred = model.predict(X_val[features], num_iteration=model.best_iteration)
    val_RMSE_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(f'val RMSE score: {val_RMSE_score}')

    # val_WRMSSE_score = evaluator.wrmsse(val_pred, y_val)
    # print(f'val RMSE score: {val_WRMSSE_score}')

    # test for submission
    y_pred = model.predict(test[features])
    test['demand'] = 1.02 * y_pred

    return test


def result_submission(test, submission_df, date):
    output_dir = "../output/"
    submission_name = "submission_{}.csv".format(date)
    submission_path = output_dir + submission_name
    os.makedirs(output_dir, exist_ok=True)

    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index='id',
                           columns='date', values='demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [
        row for row in submission_df['id'] if 'evaluation' in row]
    evaluation = submission_df[submission_df['id'].isin(evaluation_rows)]

    validation = submission_df[['id']].merge(predictions, on='id')
    final = pd.concat([validation, evaluation])
    final.to_csv(submission_path, index=False)
    print("\ncompleted process.")


def main():
    # read data
    t1 = time.time()
    date = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    calendar_df, sell_prices_df, train_df, submission_df = read_data()

    # preprocessing
    data_df, product_df = melt_and_merge(calendar_df, sell_prices_df, train_df, submission_df)  # nrows = 27500000
    data_df = encode_categorical(data_df)

    # feature engineering
    data_df = feature_engineering(data_df)
    data_df = reduce_mem_usage(data_df)

    # data split
    X_train, y_train, X_val, y_val, test = data_split(data_df)

    # evaluater
    evaluator = WRMSSEEvaluator(data_df, product_df)

    # train/validation
    test = run_lgb(X_train, y_train, X_val, y_val, test, date, evaluator)

    # submission
    result_submission(test, submission_df, date)

    t2 = time.time()
    print("\nspend time: {}[min]".format(str((t2 - t1) / 60)))


if __name__ == "__main__":
    main()

