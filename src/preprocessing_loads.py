# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:33:19 2023

@author: utente
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

from src.HESS_class import  *
# %%
nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
plt.rcParams.update({'font.size': 14, 'font.family': "Arial"})

regions = ['Abruzzo', 'Basilicata', 'Calabria', 'Campania', 'Emilia-Romagna',
           'Friuli-Venezia Giulia', 'Lazio', 'Liguria', 'Lombardia', 'Marche',
           'Molise', 'Piemonte', 'Puglia', 'Sardegna', 'Sicilia', 'Toscana',
           'Trentino-Alto Adige', 'Umbria', "Valle d'Aosta", 'Veneto']

# plt.rcParams["figure.constrained_layout.use"] = True

plt.rcParams['legend.fancybox'] = False

plt.rcParams['legend.edgecolor'] = '0.8'
# plt.rcParams['legend.frameon'] = False

plt.rcParams.update({'figure.autolayout': True})

plt.rcParams['legend.fontsize'] = 9

plt.rcParams['legend.title_fontsize'] = 10
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
    # plt.savefig(reg+".png", dpi=300)
# %%


observ = []
regs = []
for reg in tqdm(regions):
    d = HESS(reg).load()

    d["day"] = d.index.dayofyear

    observ.extend(d.groupby("day").sum().iloc[:, 0].tolist())
    regs.extend([reg]*len(d.groupby("day").sum().iloc[:, 0].tolist()))

df_kde = pd.DataFrame()
df_kde["Regions"] = regs
df_kde["Average daily load [kWh]"] = observ
# %%


region_classification = {
    'North': ['Lombardy', 'Piedmont', 'Trentino-Alto Adige', "Aosta Valley", 'Veneto', 'Liguria', 'Friuli-Venezia Giulia', 'Emilia-Romagna'],
    'Central': ['Lazio', 'Marche', 'Umbria', 'Tuscany', 'Abruzzo'],
    'South': ['Campania', 'Sicily', 'Calabria', 'Basilicata', 'Apulia', 'Molise', 'Sardinia']
}


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

df_kde["Regions"] = df_kde["Regions"].replace(italian_to_english)


plt.rcParams['legend.fontsize'] = 12

plt.rcParams['legend.title_fontsize'] = 14
# fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 9.5))


for j, i in enumerate(region_classification.keys()):
    sns.kdeplot(data=df_kde[df_kde["Regions"].isin(
        region_classification[i])], x="Average daily load [kWh]", hue="Regions", ax=axes[j])
    axes[j].set_title(f"{i} regions")


plt.suptitle("Residential end-users, power class 1.5-3.0 kW")
plt.savefig("Daily load kde.png", dpi=300)

# axes[0].set_title(f"{i} regions")

# %%
# plt.hist(d.groupby("day").sum(),density=True,edgecolor="k")
# ax=plt.gca()
# sns.kdeplot(data=d.groupby("day").sum(),x="load (kWh)",ax=ax)


df_kde_mean = df_kde.groupby("Regions").mean(
).sort_values(by="Average daily load [kWh]")


plt.figure(figsize=(8, 10))
ax = plt.gca()
sns.barplot(x="Average daily load [kWh]", y=df_kde_mean.index,
            data=df_kde_mean, palette="coolwarm",
            order=df_kde_mean.index,
            edgecolor="k",
            ax=ax)
ax.set_xlim(2, 6)
ax.set_title("Residential end-users\n power class 1.5-3.0 kW")

plt.savefig("Daily average load.png", dpi=300)


# plt.barh(df_kde_mean.index,df_kde_mean["Daily load [kWh]"],edgecolor="k")
