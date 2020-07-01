import pandas as pd
import numpy as np


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


def group_sales(grid, column):
    sum_group = grid.groupby([column, "d"]).sum()
    mean_group = grid.groupby([column, "d"]).mean()
    median_group = grid.groupby([column, "d"]).median()
    std_group = grid.groupby([column, "d"]).std()
    max_group = grid.groupby([column, "d"]).max()

    sum_group = sum_group.loc[:, "sales"].rename(column + '_sum_sales')
    mean_group = mean_group.loc[:, "sales"].rename(column + '_mean_sales')
    median_group = median_group.loc[:, "sales"].rename(column + '_median_sales')
    std_group = std_group.loc[:, "sales"].rename(column + '_std_sales')
    max_group = max_group.loc[:, "sales"].rename(column + '_max_sales')

    grid = pd.merge(grid, sum_group, on=[column, "d"])
    grid = pd.merge(grid, mean_group, on=[column, "d"])
    grid = pd.merge(grid, median_group, on=[column, "d"])
    grid = pd.merge(grid, std_group, on=[column, "d"])
    grid = pd.merge(grid, max_group, on=[column, "d"])

    grid.sort_values(["id", "d"], inplace=True)
    grid.reset_index(drop=True, inplace=True)

    return grid


def concat_grid_and_pred(grid, pred):
    pred = pred[pred["id"].str.contains("validation")]
    pred["id"] = pred["id"].str.replace("validation", "evaluation")
    pred.columns = ["id"] + [str(i) for i in range(END_TRAIN+1, END_TRAIN+29)]
    
    pred = pd.melt(pred, id_vars = "id", value_vars=pred.columns[1:], var_name = "d", value_name = "sales")
    pred["d"] = pred["d"].astype(np.int16)

    grid.sort_values(["id", "d"], inplace=True)
    pred.sort_values(["id", "d"], inplace=True)
    grid.reset_index(drop=True, inplace=True)
    pred.reset_index(drop=True, inplace=True)

    grid.loc[(END_TRAIN < grid["d"]) & (grid["d"] <= END_TRAIN + P_HORIZON), "sales"] = pred["sales"].to_numpy()
    return grid


END_TRAIN = 1941
P_HORIZON = 28

grid_df = pd.read_pickle("../input/m5-simple-fe/grid_part_1.pkl")
grid_df.sort_values(["id", "d"], inplace=True)
grid_df.reset_index(drop=True, inplace=True)

grid2 = grid_df.copy()
columns_list = ["store_id", "cat_id", "state_id"]
pred = pd.read_csv("../sub/sub_v2_store_id_0.474.csv")

grid2 = concat_grid_and_pred(grid2, pred)

for column_name in columns_list:
    grid2 = group_sales(grid2, column_name)


grid2 = reduce_mem_usage(grid2)
grid2["sales"] = grid_df["sales"]
grid2.to_pickle('../input/m5-simple-fe/grid_part_1_{}_last2.pkl'.format(str(END_TRAIN)))
