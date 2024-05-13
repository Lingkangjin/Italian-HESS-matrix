import matplotlib.pyplot as plt
import matplotlib
import os
# matplotlib.use('TkAgg')
import numpy as np
import geopandas as gpd
from geodatasets import get_path  # This is a helper function to get the path to the dataset
path = get_path("naturalearth.land")
df = gpd.read_file(path)

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from src.Geopandas import Geoplot

from src.Opt_with_load import *

# %%


plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["figure.constrained_layout.use"] = True
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)



#%%

italian_to_english = {
    'Abruzzo': 'Abruzzo',
    'Basilicata': 'Basilicata',
    'Calabria': 'Calabria',
    'Campania': 'Campania',
    'Emilia-Romagna': 'Emilia-Romagna',
    'Friuli-Venezia Giulia': 'Friuli-Venezia Giulia',
    'Lazio': 'Lazio',
    'Liguria': 'Liguria',
    'Lombardia': 'Lombardia',
    'Marche': 'Marche',
    'Molise': 'Molise',
    'Piemonte': 'Piemonte',
    'Puglia': 'Apulia',
    'Sardegna': 'Sardegna',
    'Sicilia': 'Sicily',
    'Toscana': 'Toscana',
    'Trentino-Alto Adige': 'Trentino-Alto Adige',
    'Umbria': 'Umbria',
    "Valle d'Aosta": "Valle d'Aosta",
    'Veneto': 'Veneto'
}


#
# saving_fold = os.path.join(os.getcwd(),
#                            "Results",
#                            "With load")
#
# de_tot = pd.DataFrame(columns=[["PV size (W)", "EZ size (W)", "BESS Cap [Wh]",
#                       "BESS Power [W]", "minimum lf", "Produced H2 (kg)", "minimum EZ load (W)"]])
#
# for name in italian_to_english.keys() :
#     print(name+" Started")
#     print("-----------")
#
#     if not os.path.exists(os.path.join(saving_fold,name)):
#
#         # if the demo_folder directory is not present
#         # then create it.
#         os.makedirs(os.path.join(saving_fold,name))
#     else:
#         print("Folder present")
#
#     loads, df_PV = loading(name,italian_to_english)
#
#     d = clusters(loads, df_PV)
#
#     df_summ = df_summary(d, os.path.join(saving_fold,name),name)
#
#     plt.close("all")
#
#     de_tot.loc[f"{name.split('.')[0]}"] = df_summ[df_summ['EZ size (W)'] ==
#                                                   df_summ['EZ size (W)'].min()].iloc[0, :].tolist()
#
# de_tot.to_csv(os.path.join(saving_fold,"df_all.csv"))

# %%


Geoplot()

# %%
#
# # Extract Europe
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#
# # Extract Europe
# europe = world[world['continent'] == 'Europe']
#
#
# # %%
# # Create a plot
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=ccrs.PlateCarree()))
#
# # Plot the map
# world.plot(ax=ax, color='lightblue', edgecolor='black')
#
# # Set the extent to focus on Europe
# ax.set_extent([-10, 40, 35, 70])
#
# # Add coastlines
# ax.coastlines()
#
# # Add gridlines
# ax.gridlines()
#
# # Add title
# plt.title('Europe')
#
# # Show the plot
# plt.show()



# %%

nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

plt.rcParams.update({'font.size': 14, 'font.family': "Times New Roman"})

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

# %% European map
path = "Input_data\\"

geopd = gpd.read_file(os.path.join(path+'NUTS_RG_20M_2021_4326.shp'))
# geopd =gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


plt.figure(figsize=(15, 8))
ax = plt.gca()

import matplotlib.colors as mcolors

# Define the number of colors
num_colors = 37

cmap = matplotlib.colormaps.get_cmap('viridis')

# Create an array of evenly spaced numbers between 0 and 1
indices = np.linspace(0, 1, num_colors)

# Generate the color palette by interpolating colors from the colormap
color_palette = [mcolors.to_hex(cmap(idx)) for idx in indices]


for j,i in enumerate(geopd['CNTR_CODE'].unique()):
    geopd[geopd['CNTR_CODE'] ==i].plot(ax=ax,
                                       color=color_palette[j],
                                       edgecolor='k',
                                       alpha=0.2,
                                       label=i)


#
# geopd.plot(ax=ax,
#            edgecolor="k",
#            color='lightblue',
#            column='CNTR_CODE', cmap='viridis', legend=True)
ax.set_xlim(-25, 40)
ax.set_xlabel('latitude')
ax.set_ylim(32, 75)
ax.set_ylabel('longitude')
plt.savefig('Europe_NUTS.png', dpi=300)

# plt.show()
#
#
# geopd['centroid_lon'] = geopd.centroid.x
# geopd['centroid_lat'] = geopd.centroid.y
#
# # geopd[['centroid_lon','centroid_lat']]
#
# PV_year_avail=[]
# for i in tqdm(range(len(geopd['centroid_lon']))):
#     data = PV_data(lat=geopd['centroid_lat'][i],
#                            long=geopd['centroid_lon'][i],
#                            cap=None,
#                            Hemisphere='North',
#                            year=2019).get_data().sum()
#     PV_year_avail.append(data)
#
#
# geopd['PV annual availability kWh']=PV_year_avail
#
# plt.figure(figsize=(10, 6))
# geopd.plot(column='PV annual availability kWh', cmap='viridis', alpha=0.8)
# plt.colorbar()


plt.show()
