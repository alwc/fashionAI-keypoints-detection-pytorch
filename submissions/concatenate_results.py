import sys
sys.path.append("..")

import pandas as pd

from utils import Config

opt = Config()

dfs = []
for category in ['blouse', 'skirt', 'outwear', 'dress', 'trousers']:
    df = pd.read_csv(opt.pred_path + category + '.csv')
    dfs.append(df)

res_df = pd.concat(dfs)
res_df.to_csv(opt.proj_path +'result.csv', index=False)
