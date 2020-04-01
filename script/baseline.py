import gc
import os
import warnings
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics

warnings.filterwarnings('ignore')
"""
NOTE
very fast model created by @ragnar

"""
# メモリ使用量の削減
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


# function to read the data and merge it 
# (ignoring some columns, this is a very fst model)
def read_data():
    print('Reading files...')

    calendar_df = pd.read_csv('./input/m5-forecasting-accuracy/calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns\n'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sell_prices_df = pd.read_csv('./input/m5-forecasting-accuracy/sell_prices.csv')
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns\n'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    sales_train_validation_df = pd.read_csv('./input/m5-forecasting-accuracy/sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], 
                                                                       sales_train_validation_df.shape[1]))

    submission_df = pd.read_csv('./input/m5-forecasting-accuracy/sample_submission.csv')
    return calendar_df, sell_prices_df, sales_train_validation_df, submission_df


def melt_and_merge(calendar_df, sell_prices_df, sales_train_validation_df, submission_df, nrows=55000000, merge=False):

    # 商品情報を抽出
    product_df = sales_train_validation_df.loc[:, "id":"state_id"]
    
    # 列方向に連なっていたのを変形し行方向に連ねるように整理
    print("\t[BEFORE] {} rows and {} columns".format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))
    sales_train_validation_df = pd.melt(sales_train_validation_df, 
                                        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                        var_name = 'day', value_name = 'demand')
    print('\t[AFTER] {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))
    sales_train_validation_df = reduce_mem_usage(sales_train_validation_df)
    
    # seperate test dataframes
    valid_df = submission_df[submission_df["id"].str.contains("validation")]
    eval_df = submission_df[submission_df["id"].str.contains("evaluation")]
    
    # change column names
    # validation data: F1 ~ F28 => d_1914 ~ d_1941
    valid_df.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923',
                        'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 'd_1932', 'd_1933', 
                        'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']

    # evaluation data: F1 ~ F28 => d_1942 ~ d_1969
    eval_df.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951',
                       'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 'd_1960', 'd_1961',
                       'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
    

    # melt, mergeを使ってsubmission用のdataframeを上のsales_train_validationと同様の形式に変形
    valid_df = valid_df.merge(product_df, how = 'left', on = 'id')
    valid_df = pd.melt(valid_df, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                       var_name = 'day', value_name = 'demand')

    # valid_dfと同様eval_dfとproduct_dfをmergeさせたい
    # しかしidが_evaluationのままだとデータが一致せずmergeできないので一時的に_validationにidを変更
    eval_df['id'] = eval_df.loc[:, 'id'].str.replace('_evaluation','_validation')
    eval_df = eval_df.merge(product_df, how = 'left', on = 'id')
    eval_df['id'] = eval_df.loc[:, 'id'].str.replace('_validation','_evaluation')
    eval_df = pd.melt(eval_df, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                      var_name = 'day', value_name = 'demand')

    sales_train_validation_df['part'] = 'train'
    valid_df['part'] = 'valid'
    eval_df['part'] = 'eval'
    
    data_df = pd.concat([sales_train_validation_df, valid_df, eval_df], axis = 0)
    
    # 不要なdataframeの削除
    del sales_train_validation_df, valid_df, eval_df
    
    # NOTE get only a sample for fast training
    data_df = data_df.loc[nrows:]
    
    # drop some calendar features
    calendar_df.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    
    # delete eval_df for now
    data_df = data_df[data_df['part'] != 'eval']
    
    if merge:  # dataとcalendar,sell_pricesをmergeするか?
        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
        data_df = pd.merge(data_df, calendar_df, how = 'left', left_on = ['day'], right_on = ['d'])
        data_df.drop(['d', 'day'], inplace = True, axis = 1)

        # get the sell price data (this feature should be very important)
        data_df = data_df.merge(sell_prices_df, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
        print('Our final dataset to train has {} rows and {} columns'.format(data_df.shape[0], data_df.shape[1]))
    else: 
        pass
    
    return data_df


# 欠損値補間とlabel encoding
def transform(data_df):
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        data_df[feature].fillna('unknown', inplace=True)
    
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data_df[feature] = encoder.fit_transform(data_df[feature])
    
    return data_df


# 特徴量エンジニアリング
def feature_engineering(data_df):
    print("\n[START] feature engineering ->")
    
    # rolling demand features
    # 28個分下にずらすことで時系列データの差分を取る
    data_df['lag_t28'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    data_df['lag_t29'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
    data_df['lag_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(30))

    # per a week
    data_df['rolling_mean_t7'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data_df['rolling_std_t7'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    # per a month
    data_df['rolling_mean_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())

    # per 3 month 
    data_df['rolling_mean_t90'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())

    # half year
    data_df['rolling_mean_t180'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())

    # per a month
    data_df['rolling_std_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    data_df['rolling_skew_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
    data_df['rolling_kurt_t30'] = data_df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())
    
    # price features
    data_df['lag_price_t1'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data_df['price_change_t1'] = (data_df['lag_price_t1'] - data_df['sell_price']) / (data_df['lag_price_t1'])
    data_df['rolling_price_max_t365'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data_df['price_change_t365'] = (data_df['rolling_price_max_t365'] - data_df['sell_price']) / (data_df['rolling_price_max_t365'])
    data_df['rolling_price_std_t7'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data_df['rolling_price_std_t30'] = data_df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data_df.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
    # time features
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df['year'] = data_df['date'].dt.year
    data_df['month'] = data_df['date'].dt.month
    data_df['week'] = data_df['date'].dt.week
    data_df['day'] = data_df['date'].dt.day
    data_df['dayofweek'] = data_df['date'].dt.dayofweek
    print("[FINISH] feature engineering")
    
    return data_df


# 時系列データの時はold -> train/val/test -> new とする？
def data_split(data_df):
    # going to evaluate with the last 28 days
    X_train = data_df[data_df['date'] <= '2016-03-27']
    y_train = X_train['demand']

    X_val = data_df[(data_df['date'] > '2016-03-27') & (data_df['date'] <= '2016-04-24')]
    y_val = X_val['demand']

    test = data_df[(data_df['date'] > '2016-04-24')]
    # del data_df
    # gc.collect()

    return X_train, y_train, X_val, y_val, test



def run_lgb(X_train, y_train, X_val, y_val, test):
    # define list of features
    features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek', 
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 
                'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90', 
                'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 
                'rolling_price_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']
    
    # define random hyperparammeters
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': 5046,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 10, 
        'colsample_bytree': 0.75}

    # datasetの作成
    train_set = lgb.Dataset(X_train[features], y_train)
    val_set = lgb.Dataset(X_val[features], y_val)
    # del X_train, y_train, X_val, y_val

    # train/validation
    print("\n[START] training model ->")
    model = lgb.train(params, train_set, num_boost_round=2500, early_stopping_rounds=50, 
                      valid_sets=[train_set, val_set], verbose_eval=100)
    
    val_pred = model.predict(X_val[features])
    val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(f'Our val rmse score is {val_score}')
    
    # test for submission
    y_pred = model.predict(test.loc[features])
    test['demand'] = y_pred
    return test


def result_submission(test, submission_df):
    output_dir = "./output/"
    date = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    submission_name = "submission_{}.csv".format(date)
    submission_path = output_dir + submission_name
    os.makedirs(output_dir, exist_ok=True)

    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission_df['id'] if 'evaluation' in row] 
    evaluation = submission_df[submission_df['id'].isin(evaluation_rows)]

    validation = submission_df[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv(submission_path, index = False)
    print("\ncompleted process.")


def main():
    # read data
    calendar_df, sell_prices_df, sales_train_validation_df, submission_df = read_data()

    # preprocessing
    data_df = melt_and_merge(calendar_df, sell_prices_df, sales_train_validation_df, submission_df, nrows = 27500000, merge = True)
    data_df = transform(data_df)

    # feature engineering
    data_df = feature_engineering(data_df)
    data_df = reduce_mem_usage(data_df)

    # data split
    X_train, y_train, X_val, y_val, test = data_split(data_df)

    # train/validation
    test = run_lgb(X_train, y_train, X_val, y_val, test)

    # submission
    result_submission(test, submission_df)


if __name__ == "__main__":
    main()