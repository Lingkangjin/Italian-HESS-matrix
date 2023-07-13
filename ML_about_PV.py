# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:32:19 2023

@author: utente
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
    
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from tqdm import tqdm

from kneed import KneeLocator, DataGenerator as dg
import scienceplots

from sklearn.preprocessing import StandardScaler
#%%
nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
plt.style.use(["science", "nature"])

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({"figure.dpi": "200"})
plt.rcParams["figure.constrained_layout.use"] = True


#%%
df=pd.read_csv("PV_production.csv",index_col=0)
df.dropna(axis=0,inplace=True)

date_str = '31/12/2020'
start = pd.to_datetime(date_str) - pd.Timedelta(days=365)
hourly_periods = 8784
drange = pd.date_range(start, periods=hourly_periods, freq='H')

df.index=drange

#%% remove a day
datetime_index = pd.to_datetime("29/02/2020", format="%d/%m/%Y")

df=df[df.index.date!=datetime_index.date()]

#%%

df["Days"]=df.index.date

df_grouped=pd.DataFrame()
for i,j in enumerate(df["Days"].unique()):
   df_grouped[i+1]=list(df[df["Days"]==j]["P"])
 
df_grouped.index=np.arange(1,24+1,1)   
df_grouped=df_grouped.T

#%% scikit learn

from sklearn.cluster import KMeans

scaler=StandardScaler()
X = scaler.fit_transform(df_grouped)

# cluster = KMeans(n_clusters=8)
# cluster.fit(X)
# # inert.append(cluster.inertia_)



# a=scaler.inverse_transform(cluster.cluster_centers_) 

Cluster_max=10
#%%
wcss=[]
Silhoutte = []



for i in tqdm(range(2,Cluster_max)):
    cluster = KMeans(n_clusters=i,init = 'k-means++', max_iter=300, n_init=10, random_state=0)


    cluster.fit(X)
    wcss_iter = cluster.inertia_
    wcss.append(wcss_iter)
    labels = cluster.fit_predict(X)
    
    Silhoutte.append(silhouette_score(X, labels))
    
    
# import os
# os.environ["OMP_NUM_THREADS"] = '2'
    
#%% standard scaler
number_clusters = range(2,Cluster_max)


kl = KneeLocator(number_clusters,wcss, curve="convex", direction="decreasing")

fig1,ax=plt.subplots(nrows=2, ncols=1,sharex=True)
ax[0].plot(number_clusters,wcss,marker=".",label="Inertia",color=nature_colors[0])
ax[0].axvline(x=kl.knee,color="gray",label=f"Inertia at {kl.knee} cl")

ax[0].set_ylabel('Inertia')
ax[0].legend()
# ax[0].set_xlabel('Number of clusters')
# ax[0].grid()
ax[0].tick_params(which='minor', bottom=False, top=False, right=False, left=False)




ax[1].plot(number_clusters,Silhoutte,marker="s",label="Silhoutte",color=nature_colors[1])
ax[1].legend()
ax[1].set_xlabel('Number of clusters')
# ax[1].grid()
ax[1].tick_params(which='minor', bottom=False, top=False, right=False, left=False)
ax[1].legend()
ax[1].set_ylabel("Silhoutte")

plt.xticks(range(2,Cluster_max))


plt.suptitle('scikit-learn-The Elbow test: determining K')    


#%%
cluster = KMeans(n_clusters=kl.knee,random_state=0)

y_pred = cluster.fit_predict(X)

X_back=scaler.inverse_transform(X)
X_centers=scaler.inverse_transform(cluster.cluster_centers_)

plt.figure(888)
for yi in range(kl.knee):
    plt.subplot(3, (kl.knee//3)+1, yi + 1)
    for xx in X_back[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(X_centers[yi].ravel(), "r-")
    plt.text(0.45, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes,fontsize=6)
    plt.xlim(0, 24)
   
    if yi == 1:
        plt.title("Euclidean $k$-means using scikit-learn \n and inversed back")
#%%

labels = cluster.fit_predict(X)

sil_media = silhouette_score(X, labels)
samp = silhouette_samples(X, labels)

for i in labels:
    print(f"{np.mean(samp[labels == i] > sil_media)} \n")

np.unique(labels, return_counts=True) # number of elements in each clusters


# from scipy.stats import skew, skewtest
# np.abs(skew(X))










    
# =============================================================================
#  Using tslearn package
# =============================================================================
# #%%

# X_train=TimeSeriesScalerMeanVariance().fit_transform(df_grouped) #Scaler for time series. Scales time series so that their mean (resp. standard deviation) in each dimension is mu (resp. std).



# X_reshaped = X_train.reshape(X_train.shape[0], -1) #reshaping for easy to manage
# #%%



# # X_reshaped =df_grouped.to_numpy()


# seed=0 #random state
# km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
# y_pred = km.fit_predict(X_reshaped)


# # plt.figure()
# # for yi in range(3):
# #     plt.subplot(3, 3, yi + 1)
# #     for xx in X_reshaped[y_pred == yi]:
# #         plt.plot(xx.ravel(), "k-", alpha=.2)
# #     plt.plot(km.cluster_centers_[yi].ravel(), "r-")
# #     plt.xlim(0, 24)
   
# #     if yi == 1:
# #         plt.title("Euclidean $k$-means")
        
        
        

# #%%
# wcss=[]
# Silhoutte = []



# for i in tqdm(range(2,20)):
#     km = TimeSeriesKMeans(n_clusters=i, verbose=True, random_state=seed)

#     km.fit(X_reshaped)
#     wcss_iter = km.inertia_
#     wcss.append(wcss_iter)
#     labels = km.fit_predict(X_reshaped)
    
#     Silhoutte.append(silhouette_score(X_reshaped, labels))
    

# #%%


# number_clusters = range(2,20)

# kl = KneeLocator(number_clusters,wcss, curve="convex", direction="decreasing")




# fig2,ax=plt.subplots(nrows=2, ncols=1,sharex=True)
# ax[0].plot(number_clusters,wcss,marker=".",label="Inertia",color=nature_colors[0])
# ax[0].axvline(x=kl.knee,color="gray",label=f"Inertia at {kl.knee} cl")
# ax[0].set_ylabel('Inertia')
# ax[0].legend()
# # ax[0].set_xlabel('Number of clusters')
# # ax[0].grid()
# ax[0].tick_params(which='minor', bottom=False, top=False, right=False, left=False)




# ax[1].plot(number_clusters,Silhoutte,marker="s",label="Silhoutte",color=nature_colors[1])
# ax[1].legend()
# ax[1].set_xlabel('Number of clusters')
# # ax[1].grid()
# ax[1].tick_params(which='minor', bottom=False, top=False, right=False, left=False)
# ax[1].axvline(x=(np.argmax(Silhoutte)+min(number_clusters)),color="gray",label=f"Silhoutte at {(np.argmax(Silhoutte)+min(number_clusters))} cl")
# ax[1].legend()
# ax[1].set_ylabel("Silhoutte")

# plt.xticks(range(2,20))


# plt.suptitle('The Elbow test: determining K')
# #%%
# k=kl.knee

# seed=0 #random state
# km = TimeSeriesKMeans(n_clusters=k, verbose=True, random_state=seed)
# y_pred = km.fit_predict(X_reshaped)

# plt.figure(999)

# for yi in range(kl.knee):
#     plt.subplot(3, (kl.knee//3)+1, yi + 1)
#     for xx in X_reshaped[y_pred == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(km.cluster_centers_[yi].ravel(), "r-")
#     plt.text(0.45, 0.85,'Cluster %d' % (yi + 1),
#               transform=plt.gca().transAxes,fontsize=6)
#     plt.xlim(0, 24)
   
#     if yi == 1:
#         plt.title("Euclidean $k$-means con tslearn")
        
        
   