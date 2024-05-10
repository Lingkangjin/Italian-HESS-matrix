# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:03:04 2023

@author: admin
"""

import pandas as pd


df = pd.read_csv("df_all.csv", index_col=0)


for i in df.columns:
    if i in ['PV size (W)', 'BESS Cap [Wh]', 'BESS Power [W]']:
        df[i] = df[i].astype(int)

italian_to_english = {'Abruzzo': 'Abruzzo',
                      'Basilicata': 'Basilicata',
                      'Calabria': 'Calabria',
                      'Campania': 'Campania',
                      'Emilia-Romagna': 'Emilia-Romagna',
                      'Friuli-Venezia Giulia': 'Friuli-Venezia Giulia',
                      'Lazio': 'Lazio',
                      'Liguria': 'Liguria',
                      'Lombardia': 'Lombardy',
                      'Marche': 'Marche',
                      'Molise': 'Molise',
                      'Piemonte': 'Piedmont',
                      'Puglia': 'Apulia',
                      'Sardegna': 'Sardinia',
                      'Sicilia': 'Sicily',
                      'Toscana': 'Tuscany',
                      'Trentino-Alto Adige': 'Trentino-Alto Adige',
                      'Umbria': 'Umbria',
                      "Valle d'Aosta": "Aosta Valley",
                      'Veneto': 'Veneto'}
df.rename(index=italian_to_english, inplace=True)

# Sort the DataFrame by the new index in ascending order
df.sort_index(inplace=True)

print(df.to_latex(index=True,
                  formatters={"Index": str.upper},
                  float_format="{:.2f}".format,
                  ))
