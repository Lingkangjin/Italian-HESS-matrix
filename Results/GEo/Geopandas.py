# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:15:04 2022

@author: admin
"""

from matplotlib import image
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from shapely.geometry import LineString


# %%
path = "stanford-bb489fv3314-shapefile\\"

nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

plt.rcParams.update({'font.size': 14, 'font.family': "Arial"})

Ita = gpd.read_file(path+'bb489fv3314.shp')


# %% shp file

# Chile = gpd.read_file('chl_admbndl_admALL_bcn_itos_20211008.shp')


Ita.name_1.replace("Apulia", "Puglia", inplace=True)
Ita.name_1.replace("Sicily", "Sicilia", inplace=True)

Ita.sort_values(by="name_1", inplace=True)


# %%

df_sum = pd.read_csv("df_all.csv", index_col=0)

for i in df_sum.columns:

    Ita[i] = df_sum[i].values
    data_min = Ita[i].min()
    data_max = Ita[i].max()
    plt.figure()

    ax = plt.gca()
    Ita.plot(color="white", edgecolor="k", linewidth=.5, ax=ax)
    ax.set_axis_off()

    Ita.plot(
        column=f"{i}",
        cmap='coolwarm',
        legend=True,
        legend_kwds={"label": f"{i}", "orientation": "vertical"},
        vmin=data_min, vmax=data_max,
        ax=ax
    )

    # ax.set_title("Italian Regional resouces")

    plt.savefig(f"{i}.png", dpi=300)


# plt.figure(figsize=(15, 8))

# %%

plt.figure(figsize=(15, 8))
ax = plt.gca()
Ita.plot(color="white", edgecolor="k", linewidth=0.5, ax=ax)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")


region_classification = {
    'North': ['Lombardia', 'Piemonte', 'Trentino-Alto Adige', "Valle d'Aosta", 'Veneto', 'Liguria', 'Friuli-Venezia Giulia', 'Emilia-Romagna'],
    'Central': ['Lazio', 'Marche', 'Umbria', 'Toscana', 'Abruzzo'],
    'South': ['Campania', 'Sicilia', 'Calabria', 'Basilicata', 'Puglia', 'Molise', 'Sardegna']
}


inverted_region_classification = {region: region_type for region_type,
                                  regions in region_classification.items() for region in regions}

Ita["Zone"] = Ita.apply(
    lambda x: inverted_region_classification[x['name_1']], axis=1)

Ita.apply(lambda x: ax.annotate(
    text=x['name_1'], xy=x.geometry.centroid.coords[0], fontsize=6, ha='center'), axis=1)


Ita[Ita["Zone"] == "North"].plot(
    color=nature_colors[0], markersize=180, ax=ax,  alpha=0.8)
Ita[Ita["Zone"] == "Central"].plot(
    color=nature_colors[1], markersize=180, ax=ax, alpha=0.8)
Ita[Ita["Zone"] == "South"].plot(
    color=nature_colors[2], markersize=180, ax=ax, alpha=0.8)

ax.set_axis_off()

plt.savefig("Italian_map.png", dpi=300)
# ax.legend(fancybox=False, shadow=False, edgecolor="k", borderpad=.3, loc=2)

# color = iter(cm.tab20(np.linspace(0, 1, len(gdf))))


# for i, j in enumerate(df.Sites):
#     c = next(color)
#     gdf[gdf.Sites == j].plot(ax=ax, markersize=180, c=c, marker='.', label=j)

# # %%
# lines = []

# for i in range(len(gdf) - 1):
#     line = LineString([gdf.geometry.iloc[i], gdf.geometry.iloc[i + 1]])
#     lines.append(line)

# lines_gdf = gpd.GeoDataFrame({'geometry': lines}, crs=gdf.crs)

# lines_gdf.plot(ax=ax, color='red', linewidth=2)
# # Add arrows
# for idx, row in gdf.iterrows():
#     if idx < len(gdf) - 1:
#         x0, y0 = row['geometry'].x, row['geometry'].y
#         x1, y1 = gdf.iloc[idx +
#                           1]['geometry'].x, gdf.iloc[idx + 1]['geometry'].y
#         ax.annotate(
#             '',
#             xy=(x1, y1), xycoords='data',
#             xytext=(x0, y0), textcoords='data',
#             arrowprops=dict(arrowstyle="->", color='black', linewidth=2),
#         )
# # %%
# # gdf.plot(ax=ax,color='red',markersize=20,marker='o',label="legend")

# ax.set_title("Chile-Antofagasta map")
# ax.set_axis_off()


# ax.legend(title="All sites", bbox_to_anchor=(
#     0.95, 1), edgecolor="k", fancybox=False)

# # plt.colorbar(ax)


# plt.tight_layout()
# plt.savefig("Chile tot map.png",dpi=300)
# b=world[world.name == 'Chile']
# CHL.plot(
#     color='white', edgecolor='black')
