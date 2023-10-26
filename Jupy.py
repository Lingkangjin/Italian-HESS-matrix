# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:57:30 2023

@author: utente
"""

import numpy as np
from ridgeplot import ridgeplot
from ridgeplot.datasets import load_lincoln_weather
import matplotlib.pyplot as plt
import subprocess


plt.rcParams.update({'font.size': 14, 'font.family': "Arial"})

df = load_lincoln_weather()

# Transform the data into a 3D (ragged) array format of
# daily min and max temperature samples per month
months = df.index.month_name().unique()
samples = [
    [
        df[df.index.month_name() == month]["Min Temperature [F]"],
        df[df.index.month_name() == month]["Max Temperature [F]"],
    ]
    for month in months
]

# And finish by styling it up to your liking!
fig = ridgeplot(
    samples=samples,
    labels=months,
    coloralpha=0.98,
    bandwidth=4,
    kde_points=np.linspace(-25, 110, 400),
    spacing=0.33,
    linewidth=2,
)
fig.update_layout(
    title="Minimum and maximum daily temperatures in Lincoln, NE (2016)",
    height=650,
    width=950,
    font_family="Arial",
    font_size=14,
    plot_bgcolor='white',
    # plot_bgcolor="rgb(245, 245, 245)",
    xaxis_gridcolor="white",
    yaxis_gridcolor="white",
    xaxis_gridwidth=2,
    yaxis_title="Month",
    xaxis_title="Temperature [F]",
    showlegend=False,
)


fig.write_html("file.html")
subprocess.Popen(["file.html"], shell=True)
