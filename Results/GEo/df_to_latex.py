# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:07:14 2023

@author: admin
"""

import pandas as pd


df = pd.read_csv("df_all.csv", index_col=0)
df = df.drop(columns=["Minimum lf"])

print(df.to_latex(index=True,
                  formatters={"Index": str.upper},
                  float_format="{:.2f}".format,
                  ))
