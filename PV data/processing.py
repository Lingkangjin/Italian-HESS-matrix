# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:00:14 2023

@author: utente
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
# %%

inp = "Raw\\"
out = "Processed\\"
plt.rcParams.update({'legend.fontsize': 8, 'legend.title_fontsize': 12,
                    'font.size': 12, 'font.family': "Arial"})
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams['legend.frameon'] = False

nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
# %%
names = []
for path, subdirs, files in os.walk(inp):
    for name in files:
        if name.endswith('.csv'):
            print(name)
            names.append(name)
    else:
        pass

# %%


dictionary = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

df_all = pd.DataFrame(index=dictionary.values())

yearly = []
indexes = []
for name in names:
    df = pd.read_csv(inp+name, skiprows=3, index_col=0,
                     parse_dates=True)["electricity"]
    # df.to_csv(out+i)

    PV_month = []

    months = []

    for i in df.index.month.unique():
        months.append(dictionary[i])
        PV_month.append(df[df.index.month == i].sum()/(24*30))

    df_all[name.split(".")[0]] = PV_month
    indexes.append(name.split(".")[0])
    yearly.append(round(df.sum()/8760, 3))


df_year = pd.DataFrame(index=indexes)
df_year["$C_f$"] = yearly


def split_list(lst, chunk_size):
    return list(zip(*[iter(lst)] * chunk_size))

# # plots
# plt.bar(months, [i/1000 for i in PV_month], edgecolor="k")
# plt.ylabel("kWh")
# plt.xticks(rotation=30)
# plt.title("Monthly production distribution")


a = split_list(df_all.columns, 3)


fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True,
                         sharey=True, figsize=(13, 8))
fig.delaxes(axes[2, 2])
fig.delaxes(axes[2, 1])


df_all[list(a[0])].plot.bar(edgecolor="k",  ax=axes[0, 0], alpha=0.8)
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
# axes[0, 0].legend(loc='center left', bbox_to_anchor=(
#     1, 0.5), fancybox=False, edgecolor="k", title="Regions")
axes[0, 0].set_ylabel("$C_f$")
axes[0, 0].set_ylim(0, 0.28)


df_all[list(a[1])].plot.bar(edgecolor="k",  ax=axes[0, 1], alpha=0.8)
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
axes[0, 1].set_ylabel("$C_f$")


df_all[list(a[2])].plot.bar(edgecolor="k",  ax=axes[0, 2], alpha=0.8)
axes[0, 2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
axes[0, 2].set_ylabel("$C_f$")


df_all[list(a[3])].plot.bar(edgecolor="k",  ax=axes[1, 0], alpha=0.8)
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
axes[1, 0].set_ylabel("$C_f$")

df_all[list(a[4])].plot.bar(edgecolor="k", ax=axes[1, 1], alpha=0.8)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
axes[1, 1].set_ylabel("$C_f$")
axes[1, 1].set_xticklabels(
    axes[1, 1].get_xticklabels(), rotation=45, ha='right')


df_all[list(a[5])].plot.bar(edgecolor="k", ax=axes[1, 2], alpha=0.8)
axes[1, 2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
axes[1, 2].set_ylabel("$C_f$")
axes[1, 2].set_xticklabels(
    axes[1, 2].get_xticklabels(), rotation=45, ha='right')


df_all[list(df_all.columns[-2:])].plot.bar(edgecolor="k",
                                           ax=axes[2, 0], alpha=0.8)
axes[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                  ncol=3, fancybox=False)
axes[2, 0].set_ylabel("$C_f$")
axes[2, 0].set_xticklabels(
    axes[2, 0].get_xticklabels(), rotation=45, ha='right')

plt.savefig("Italian monthly average prod.png", dpi=300)


# %%

region_classification = {
    'North': ['Lombardia', 'Piemonte', 'Trentino-Alto Adige', "Valle d'Aosta", 'Veneto', 'Liguria', 'Friuli-Venezia Giulia', 'Emilia-Romagna'],
    'Central': ['Lazio', 'Marche', 'Umbria', 'Toscana', 'Abruzzo'],
    'South': ['Campania', 'Sicilia', 'Calabria', 'Basilicata', 'Puglia', 'Molise', 'Sardegna']
}


df_year.loc[region_classification['North']]

plt.rcParams.update({'legend.fontsize': 14})


fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True)

for i, j in enumerate(region_classification.keys()):
    df_year.loc[region_classification[j]].plot.barh(
        edgecolor="k", color=nature_colors[i], ax=ax[i])

    for container in ax[i].containers:
        ax[i].bar_label(container, fmt='%.2f', fontsize="11")
    ax[i].legend([j])
    ax[i].set_xlim(0.155, 0.20)
    ax[i].set_xlabel("Yearly capacity factor")

plt.suptitle("Regional PV capacity factor")
plt.savefig("PV_capacity factor.png", dpi=300)
