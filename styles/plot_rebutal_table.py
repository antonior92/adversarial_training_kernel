import pandas as pd
import numpy as np
import re

# Load CSV
df1 = pd.read_csv("out/performance_rebutall.csv")
df2 = pd.read_csv("out/performance_regr_short.csv")
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['method'] != 'amkl']
df = df[df['dset'] == 'polution']

# Normalize column types
df["adv"] = df["adv"].astype(bool)
df["p"] = df["p"].replace("inf", np.inf).astype(float)

index = df[df['adv'] ==False]['method'].values


col1 = df[df['adv'] ==False]['r2_score']
col5 = df[(df['adv']==True) & (df['radius'] == 0.1) & (df['p']==np.inf)]['r2_score']

result_df = pd.DataFrame(np.array([np.array(col1), np.array(col5)]).T, index=index)

print(result_df.round(2).to_markdown())