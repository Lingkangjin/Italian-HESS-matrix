# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:15:04 2022

@author: admin
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# %%


path=os.path.join(os.getcwd(),
                  "Input_data",
                  "stanford-bb489fv3314-shapefile",
                  'bb489fv3314.shp')


nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']




# %%

def Geoplot():
    Ita = gpd.read_file(path)

    Ita.name_1.replace("Apulia", "Puglia", inplace=True)
    Ita.name_1.replace("Sicily", "Sicilia", inplace=True)

    Ita.sort_values(by="name_1", inplace=True)
    df_sum= pd.read_csv(os.path.join(os.getcwd(),
                                     "Results",
                                     "With loaddf_all.csv"),
                        index_col=0)


    column_mapping = {
        'PV size (W)': 'PV size [W]',
        'EZ size (W)': 'Electrolyser size [W]',
        'BESS Cap (Wh)': 'Battery Capacity [Wh]',
        'BESS Power (W)': 'Battery Power [W]',
        'Minimum EZ load (W)': 'Minimum Electrolyser load [W]',
        'Produced $H_2$ (kg)': "Daily produced $H_2$ [kg]"
    }

    # Rename the columns based on the mapping
    df_sum = df_sum.rename(columns=lambda col: column_mapping.get(col, col))

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
        plt.tight_layout()
        plt.savefig(os.path.join('Results',
            f"{i}.png"), dpi=300)


    # plt.figure(figsize=(15, 8))

    # %%
    Ita.name_1.replace('Piemonte', 'Piedmont', inplace=True)
    Ita.name_1.replace('Lombardia', 'Lombardy', inplace=True)
    Ita.name_1.replace('Sardegna', 'Sardinia', inplace=True)

    Ita.name_1.replace("Puglia", "Apulia", inplace=True)
    Ita.name_1.replace("Sicilia", "Sicily", inplace=True)
    Ita.name_1.replace('Toscana', 'Tuscany', inplace=True)
    Ita.name_1.replace("Valle d'Aosta", "Aosta Valley", inplace=True)


    region_classification = {
        'North': ['Lombardy', 'Piedmont', 'Trentino-Alto Adige', "Aosta Valley", 'Veneto', 'Liguria', 'Friuli-Venezia Giulia', 'Emilia-Romagna'],
        'Central': ['Lazio', 'Marche', 'Umbria', 'Tuscany', 'Abruzzo'],
        'South': ['Campania', 'Sicily', 'Calabria', 'Basilicata', 'Apulia', 'Molise', 'Sardinia']
    }

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    Ita.plot(color="white", edgecolor="k", linewidth=0.5, ax=ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")



    inverted_region_classification = {region: region_type for region_type,
                                      regions in region_classification.items() for region in regions}

    Ita["Zone"] = Ita.apply(
        lambda x: inverted_region_classification[x['name_1']], axis=1)

    Ita.apply(lambda x: ax.annotate(
        text=x['name_1'], xy=x.geometry.centroid.coords[0], fontsize=9, ha='center'), axis=1)


    Ita[Ita["Zone"] == "North"].plot(
        color=nature_colors[0], markersize=180, ax=ax,  alpha=0.8)
    Ita[Ita["Zone"] == "Central"].plot(
        color=nature_colors[1], markersize=180, ax=ax, alpha=0.8)
    Ita[Ita["Zone"] == "South"].plot(
        color=nature_colors[2], markersize=180, ax=ax, alpha=0.8)

    ax.set_axis_off()

    plt.savefig(os.path.join('Results', "Italian_map.png"), dpi=300)

