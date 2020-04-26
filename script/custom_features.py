import os
import numpy as np
import pandas as pd


def main():
    TARGET = "sales"

    grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')
    grid_df[TARGET][grid_df['d']>(1913-28)] = np.nan
    base_cols = list(grid_df)

    icols =  [['state_id'],
              ['store_id'],
              ['cat_id'],
              ['dept_id'],
              ['state_id', 'cat_id'],
              ['state_id', 'dept_id'],
              ['store_id', 'cat_id'],
              ['store_id', 'dept_id'],
              ['item_id'],
              ['item_id', 'state_id'],
              ['item_id', 'store_id']]

    for col in icols:
        print('Encoding', col)
        col_name = '_'+'_'.join(col)+'_'
        grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)
        grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)

    keep_cols = [col for col in list(grid_df) if col not in base_cols]
    grid_df = grid_df[['id','d']+keep_cols]

    print('Save Mean/Std encoding')
    save_path = '../input/m5-custom-features/'
    os.makedirs(save_path, exist_ok=True)
    grid_df.to_pickle(save_path + 'mean_encoding_df.pkl')


if __name__ == "__main__":
    main()
