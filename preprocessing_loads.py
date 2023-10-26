# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:33:19 2023

@author: utente
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from HESS_reg_class import *
# %%
nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
plt.rcParams.update({'font.size': 14, 'font.family': "Arial"})

regions = ['Abruzzo', 'Basilicata', 'Calabria', 'Campania', 'Emilia-Romagna',
           'Friuli-Venezia Giulia', 'Lazio', 'Liguria', 'Lombardia', 'Marche',
           'Molise', 'Piemonte', 'Puglia', 'Sardegna', 'Sicilia', 'Toscana',
           'Trentino-Alto Adige', 'Umbria', "Valle d'Aosta", 'Veneto']

# %%
d = HESS("Abruzzo").load()

dict_Abruzzo = {}
for i in np.arange(1, 13, 1):
    de = d[d.index.month == i]
    df_month = pd.DataFrame(index=np.arange(24))

    for j in de["Days"].unique():
        df_month[j] = de[de["Days"] == j]['load (kWh)'].tolist()[:24]
    dict_Abruzzo[i] = df_month


# from ridgeplot import ridgeplot
# from ridgeplot.datasets import load_lincoln_weather
# import matplotlib.pyplot as plt
# import subprocess


# months = d.index.month_name().unique()

# samples = [
#     [
#         d[(d.index.month_name() == month) & (d["Days"] == "Working day")]['load (kWh)'][:24],
#         d[(d.index.month_name() == month) & (d["Days"] == "Saturday")]['load (kWh)'][:24],
#         d[(d.index.month_name() == month) & (d["Days"] == "Sunday")]['load (kWh)'][:24]

#     ]
#     for month in months
# ]


# fig = ridgeplot(
#     samples=samples,
#     labels=months,
#     coloralpha=0.98,
#     bandwidth=4,
#     kde_points=np.linspace(-25, 110, 400),
#     spacing=0.33,
#     linewidth=2,
# )
# fig.update_layout(
#     title="2021 Italian Abruzzo <3 kW residential load: Weekday/Saturday/Sunday",
#     height=650,
#     width=950,
#     font_family="Arial",
#     font_size=14,
#     plot_bgcolor='white',
#     # plot_bgcolor="rgb(245, 245, 245)",
#     xaxis_gridcolor="white",
#     yaxis_gridcolor="white",
#     xaxis_gridwidth=2,
#     yaxis_title="Month",
#     xaxis_title="kWh",
#     showlegend=False,
# )


# fig.write_html("file.html")
# subprocess.Popen(["file.html"], shell=True)


# %%

for reg in tqdm(regions):
    d = HESS(reg).load()

    dict_Abruzzo = {}
    for i in np.arange(1, 13, 1):
        de = d[d.index.month == i]
        df_month = pd.DataFrame(index=np.arange(24))

        for j in de["Days"].unique():
            df_month[j] = de[de["Days"] == j]['load (kWh)'].tolist()[:24]
        dict_Abruzzo[i] = df_month

    month_dict = dict(zip(np.arange(1, 13, 1), d.index.month_name().unique()))

    fig, axes = plt.subplots(ncols=4, nrows=3, sharex=True,
                             sharey=True, figsize=(10, 8))

    for i in range(3):
        for j in range(4):
            key = 1+i*4+j
            for color, column in enumerate(dict_Abruzzo[key].columns):
                dict_Abruzzo[key][column].plot.area(
                    alpha=0.35, label=column, ax=axes[i, j], color=nature_colors[color])
            axes[i, j].set_title(month_dict[key])
            axes[i, j].set_ylabel("kWh")
            if i == 1 and j == 3:
                box = axes[i, j].get_position()
                # axes[i, j].set_position([box.x0, box.y0, box.width * 0.8, box.height])

                # Put a legend to the right of the current axis
                axes[i, j].legend(loc='center left', bbox_to_anchor=(
                    1, 0.5), fancybox=False, edgecolor="k")
            # axes[i, j].legend(fancybox=False,edgecolor="k")

    plt.suptitle(reg+" residential load: power class <3 kW")
    plt.savefig(reg+".png", dpi=300)
