import matplotlib.pyplot as plt
import matplotlib
import os
# matplotlib.use('TkAgg')
import numpy as np
import geopandas as gpd
from geodatasets import get_path  # This is a helper function to get the path to the dataset
path = get_path("naturalearth.land")
df = gpd.read_file(path)

from src.PV_data_request import PV_data
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from src.Geopandas import Geoplot

#%%


path=os.path.join(os.getcwd(),
                  "Input_data",
                  "stanford-bb489fv3314-shapefile",
                  'bb489fv3314.shp')





Ita = gpd.read_file(path)


data=PV_data(lat=Ita[Ita.name_1=='Apulia'].centroid.x,
            long=Ita[Ita.name_1=='Apulia'].centroid.y,
            cap=None,
            Hemisphere='North',
            year=2019).get_data()


# Geoplot()
ok

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

plt.rcParams.update({'font.size': 14, 'font.family': "Arial"})

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

# %%
path = "Input_data\\"

geopd = gpd.read_file(os.path.join(path+'NUTS_RG_20M_2021_4326.shp'))
# geopd =gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# %%
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
ax.set_ylim(32, 75)
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
