# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:07:55 2023

@author: utente
"""

from sklearn.cluster import KMeans
import numpy as np


from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from tqdm import tqdm

from kneed import KneeLocator, DataGenerator as dg

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams.update({'font.size': 10})

nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']


# %%
scaler = StandardScaler()


class ML_cluser_TS():

    def __init__(self, df):
        self.df = df  # it has columns different demands

    def finding_n_cluster(self, plotting):

        # initialization

        X = scaler.fit_transform(self.df)

        Cluster_max = 10

        wcss = []
        Silhoutte = []

        for i in tqdm(range(2, Cluster_max)):
            cluster = KMeans(n_clusters=i, init='k-means++',
                             max_iter=300, n_init=10, random_state=0)

            cluster.fit(X)
            wcss_iter = cluster.inertia_
            wcss.append(wcss_iter)
            labels = cluster.fit_predict(X)
            Silhoutte.append(silhouette_score(X, labels))

        # plotting
        number_clusters = range(2, Cluster_max)

        kl = KneeLocator(number_clusters, wcss, curve="convex",
                         direction="decreasing")

        if plotting == "YES":
            fig1, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax[0].plot(number_clusters, wcss, marker=".",
                       label="Inertia", color=nature_colors[0])
            ax[0].axvline(x=kl.knee, color="gray",
                          label=f"Inertia at {kl.knee} cl")

            ax[0].set_ylabel('Inertia')
            ax[0].legend()
            # ax[0].set_xlabel('Number of clusters')
            # ax[0].grid()
            ax[0].tick_params(which='minor', bottom=False,
                              top=False, right=False, left=False)

            ax[1].plot(number_clusters, Silhoutte, marker="s",
                       label="Silhoutte", color=nature_colors[1])
            ax[1].legend()
            ax[1].set_xlabel('Number of clusters')
            # ax[1].grid()
            ax[1].tick_params(which='minor', bottom=False,
                              top=False, right=False, left=False)
            ax[1].legend()
            ax[1].set_ylabel("Silhoutte")

            plt.xticks(range(2, Cluster_max))

            plt.suptitle('scikit-learn-The Elbow test: determining K')
        else:
            pass

        self.knee = kl.knee
        self.X = X
        return

    def knee_cluster(self, plotting, output, name):
        X = scaler.fit_transform(self.df)

        cluster = KMeans(n_clusters=self.knee, random_state=0)

        y_pred = cluster.fit_predict(X)

        X_back = scaler.inverse_transform(X)
        X_centers = scaler.inverse_transform(cluster.cluster_centers_)

        if plotting == "YES":
            plt.figure(figsize=(5, 2))
            for yi in range(self.knee):
                if self.knee//3 > 1:
                    # plt.figure(888, figsize=(8, 5*self.knee/3))
                    plt.subplot(3, (self.knee//3)+1, yi+1)
                else:
                    # plt.figure(888, figsize=(8, 5*self.knee/3))
                    plt.subplot(1, self.knee, yi+1)
                for xx in X_back[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(X_centers[yi].ravel(), "r-")
                plt.text(0.45, 0.85, 'Cluster %d' % (yi + 1),
                         transform=plt.gca().transAxes, fontsize=10)
                # plt.xlim(0, 24)

                # if yi == 1:
                #     plt.title("Euclidean $k$-means using scikit-learn \n and inversed back")
                if output == "YES":
                    plt.suptitle(name+" outputs $k$-means")
                else:
                    plt.suptitle("$k$-means")
        else:
            pass

        self.cluster = cluster
        return X_centers, X_back, y_pred

    def checking_silhoutte(self):
        X = scaler.fit_transform(self.df)

        cluster = KMeans(n_clusters=self.knee, random_state=0)

        labels = cluster.fit_predict(X)

        sil_media = silhouette_score(X, labels)
        samp = silhouette_samples(X, labels)

        # for i in labels:
        #     print(f"{np.mean(samp[labels == i] > sil_media)} \n")

        # number of elements in each clusters
        return np.unique(labels, return_counts=True)
