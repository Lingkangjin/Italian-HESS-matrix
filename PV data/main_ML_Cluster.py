# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:26:22 2023

@author: utente
"""

from PV_ML import *
from tqdm import tqdm
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# %%
Clu = "Clusters\\"
Proc = "Processed\\"


names = []
for path, subdirs, files in os.walk(Proc):
    for name in files:
        if name.endswith('.csv'):
            print(name)
            names.append(name)
    else:
        pass

# %%

for name in tqdm(names):
    if not os.path.exists(Clu+f"{name.split('.')[0]}"):

        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(Clu+f"{name.split('.')[0]}")
    else:
        print("Folder present")

    df = pd.read_csv(Proc+name, index_col=0, parse_dates=True)
    df["Days"] = df.index.date

    df_grouped = pd.DataFrame()
    for i, j in enumerate(df["Days"].unique()):
        df_grouped[i+1] = list(df[df["Days"] == j][df.columns[0]])

    df_grouped.index = np.arange(1, 24+1, 1)
    df_grouped = df_grouped.T

    # %%

    a = ML_cluser_TS(df_grouped)
    a.finding_n_cluster("False")
    c, back, y = a.knee_cluster("YES", "YES", name.split(".")[0])
    plt.savefig(
        Clu+f"{name.split('.')[0]}\\{name.split('.')[0]}_cluster.png", dpi=300)
    #
    df = pd.DataFrame()

    for i in range(a.knee):
        df[f"cluster {i}"] = c[i]

    df.to_csv(Clu+f"{name.split('.')[0]}\\{name.split('.')[0]}_clusters.csv")
