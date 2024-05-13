# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:26:01 2023

@author: Lingkang Jin
"""


from pyomo.environ import *
from pyomo import environ as pyo

from src.HESS_class import HESS

import os
import pandas as pd
import geopandas as gpd


from sklearn.cluster import KMeans
import numpy as np

from src.PV_data_request import PV_data

from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm import tqdm

from kneed import KneeLocator

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %%
#
nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                 '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
#
#
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["figure.constrained_layout.use"] = True
# params = {'mathtext.default': 'regular'}
# plt.rcParams.update(params)


# %%
out = os.path.join(os.getcwd(),
                   "PV data",
                    "Processed")


path=os.path.join(os.getcwd(),
                  "Input_data",
                  "stanford-bb489fv3314-shapefile",
                  'bb489fv3314.shp')





Ita = gpd.read_file(path)


# %%





# %%


def loading(name,italian_to_english):
    """
    Load data for a specific region and generate a dataframe with load information.

    Parameters:
        name (str): The name of the region for which the data is loaded.

    Returns:
        tuple: A tuple containing two dataframes. The first dataframe contains load information and the second dataframe contains PV data.

    """


    loads = HESS(name).load()
    loads["Day"] = loads.index.day_of_year

    english_name_region=italian_to_english[name.split(".")[0]]

    # using Renewable Ninja API and optimal azimuth and titl angle
    df_PV = PV_data(lat=Ita[Ita.name_1 == english_name_region].centroid.y,
                   long=Ita[Ita.name_1 == english_name_region].centroid.x,
                   cap=None,
                   Hemisphere='North',
                   year=2019).get_data()


    df_PV["Day"] = df_PV.index.day_of_year

    return loads, df_PV


# %%

scaler = StandardScaler()


class ML_cluser_TS():

    def __init__(self, df):
        self.df = df  # it has columns different demands

    def finding_n_cluster(self, plotting):
        """
            Find the optimal number of clusters using the Elbow method and silhouette score.

            Parameters:
                plotting (str): Determines whether to plot the results or not. If set to "YES", the results will be plotted. Otherwise, no plot will be generated.

            Returns:
                None

            Notes:
                - The method calculates the within-cluster sum of squares (WCSS) and silhouette score for different numbers of clusters.
                - The Elbow method is used to determine the optimal number of clusters based on the WCSS.
                - The silhouette score is also calculated to provide additional insight into the clustering quality.
                - If plotting is set to "YES", the method will generate a plot showing the WCSS and silhouette score for different numbers of clusters.
                - The optimal number of clusters (knee point) is stored in the 'knee' attribute of the class.
                - The scaled data (X) is stored in the 'X' attribute of the class.
        """

        # initialization

        X = scaler.fit_transform(self.df)

        Cluster_max = 8

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

    def knee_cluster(self, plotting):
        """
        Perform k-means clustering on the data using the optimal number of clusters determined by the 'finding_n_cluster' method.

        Parameters:
            plotting (str): Determines whether to plot the results or not. If set to "YES", the results will be plotted. Otherwise, no plot will be generated.

        Returns:
            tuple: A tuple containing three arrays. The first array contains the centroids of the clusters, the second array contains the original unscaled data, and the third array contains the predicted cluster labels for each data point.

        Notes:
            - The method first scales the data using the 'scaler' object.
            - The 'knee' attribute of the class is used to determine the number of clusters.
            - The k-means clustering algorithm is then applied to the scaled data using the optimal number of clusters.
            - The unscaled data is obtained by inverting the scaling transformation.
            - If plotting is set to "YES", the method will generate a plot showing the original data points and the centroids for each cluster.
            - The predicted cluster labels for each data point are stored in the 'y_pred' attribute of the class.
            - The centroids of the clusters are stored in the 'X_centers' attribute of the class.
            - The original unscaled data is stored in the 'X_back' attribute of the class.
        """
        X = scaler.fit_transform(self.df)

        cluster = KMeans(n_clusters=self.knee, random_state=0)

        y_pred = cluster.fit_predict(X)


        # getting centroids value and original values
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

                plt.suptitle("$k$-means")

        else:
            pass

        self.cluster = cluster
        return X_centers, X_back, y_pred



# %%

def clusters(loads, df_PV):
    """
    Perform clustering analysis on the given loads and PV data.

    Parameters:
        loads (DataFrame): A DataFrame containing load information.
        df_PV (DataFrame): A DataFrame containing PV data.

    Returns:
        dict: A dictionary containing the clustered data. The keys of the dictionary represent the cluster labels, and the values are DataFrames containing the load and PV data for each cluster.

    Notes:
        - The function first regroups the PV and loads data by day.
        - It then applies k-means clustering to the regrouped data.
        - The optimal number of clusters is determined using the 'finding_n_cluster' and 'knee_cluster' methods of the 'ML_cluser_TS' class.
        - For each cluster, the function selects the corresponding days from the original data and calculates the average load and PV values for each hour.
        - The resulting clustered data is returned as a dictionary, where each key represents a cluster label and the corresponding value is a DataFrame containing the average load and PV values for each hour.
    """
    df_regroup = pd.DataFrame(index=np.arange(1, 365+1, 1))

    # grouping the PV and loads by the
    df_regroup["PV"] = df_PV.groupby("Day").sum()[df_PV.columns[0]].values

    df_regroup["load"] = loads.groupby("Day").sum()["load (kWh)"].values

    K_means_ML = ML_cluser_TS(df_regroup)
    K_means_ML.finding_n_cluster("YES")
    X_centers, X_back, y_pred = K_means_ML.knee_cluster("YES")

    # %%
    Clusterds_dict = {}

    for i in range(K_means_ML.knee):
        index = loads["Day"].unique()[y_pred == i]
        loads["Hour"] = loads.index.hour
        df_PV["Hour"] = df_PV.index.hour

        ll = loads.set_index("Day")
        e = ll.loc[index]
        e.drop(columns=['Days']).groupby("Hour").mean()

        p = df_PV.set_index("Day")
        gen = p.loc[index]
        p.groupby("Hour").mean()

        Clusterds_dict[f"Cluster {i}"] = pd.concat(
            [e.drop(columns=['Days']).groupby("Hour").mean(), p.groupby("Hour").mean()], axis=1)

    return Clusterds_dict


# %%

def pyomo_model(lf, hlist, df):
    """
    Perform optimization modeling using Pyomo.

    Parameters:
        lf (float): The load factor.
        hlist (list): A list of hours.
        df (DataFrame): A DataFrame containing PV and load data.

    Returns:
        model: The Pyomo model object.

    """

    # parameters
    C_rate_bess_max = 1
    BESS_scale = 50  # maximum scale of power compared to PV size
    # hlist = [i for i in range(days*24)]

    eta_EZ = 0.7
    eta_BESS = 0.9

    initial_soc = 0.2
    SOC_min = 0.2
    M = 1000  # Big M value method to transfer the

    # hlist = [i for i in range(days*24)]  # hours

    PV_gen = (df[df.columns[1]]*1000).tolist()
    load = (df['load (kWh)']*1000).tolist()

    model = ConcreteModel()  # Pyomo concrete model

    # Model Dispatch Decisions
    model.PV_size = Var(within=NonNegativeReals, initialize=0)
    model.EZ_size = Var(within=NonNegativeReals, initialize=0)
    model.BESS_cap = Var(within=NonNegativeReals, initialize=0)
    model.BESS_power = Var(within=NonNegativeReals, initialize=0)

    # model Timeseries Variables
    model.EZ_E_in = Var(hlist, within=NonNegativeReals, initialize=0)  # kW
    model.EZ_E_out = Var(hlist, within=NonNegativeReals, initialize=0)  # kW

    # model.EZ_on=Var(hlist, within=Boolean, initialize=1) # 1 or 0

    model.BESS_in = Var(hlist, within=NonNegativeReals, initialize=0)  # kW
    model.BESS_out = Var(hlist, within=NonNegativeReals, initialize=0)  # kW
    model.BESS_st = Var(hlist, within=NonNegativeReals, initialize=0)  # kW

    # Binary variable to represent charging status
    model.BESS_charge = Var(hlist, within=Binary, initialize=0)
    # Binary variable to represent discharging status
    model.BESS_discharge = Var(hlist, within=Binary, initialize=0)
    model.BESS_idle = Var(hlist, within=Binary,
                          initialize=1)  # Binary idle status

    # def elec_balance(model,h):
    #     return (PV_gen[h]+
    #             model.BESS_out[h]*eta_BESS-
    #             model.BESS_in[h]-
    #             model.EZ_E_in[h]
    #             ==0)

    def elec_balance(model, h):
        return (model.PV_size*PV_gen[h] +
                model.BESS_out[h]*eta_BESS*(1 - model.BESS_charge[h]) -
                model.BESS_in[h]*eta_BESS*model.BESS_charge[h] -
                model.EZ_E_in[h] -
                load[h]
                == 0)
    model.el_bal_const = Constraint(hlist, rule=elec_balance)

    model.storage_const = ConstraintList()
    for h in hlist:
        model.storage_const.add(model.BESS_st[h] <= model.BESS_cap)
        model.storage_const.add(model.BESS_in[h] <= model.BESS_power)  # power
        model.storage_const.add(model.BESS_out[h] <= model.BESS_power)

        model.storage_const.add(
            SOC_min * model.BESS_cap <= model.BESS_st[h])  # Min SOC

        model.storage_const.add(model.BESS_in[h] <= M * model.BESS_charge[h])
        model.storage_const.add(
            model.BESS_out[h] <= M * model.BESS_discharge[h])

        model.storage_const.add(model.EZ_E_in[h] <= model.EZ_size)

        model.storage_const.add(
            model.EZ_size*lf <= model.EZ_E_in[h])  # 80-100% load

    # Add a new constraint to enforce the relationship between BESS_discharge and BESS_out (correcting behaviour)
        model.storage_const.add(
            model.BESS_discharge[h] <= M * model.BESS_out[h])

    """
    C-rate maxium, causing problems
    """
    def BESS_power_energy(model):
        return (model.BESS_cap) <= model.BESS_power*C_rate_bess_max
    model.BESS_power_const = Constraint(rule=BESS_power_energy)

    # def BESS_power(model):
    #     return model.BESS_power <= 1000*BESS_scale
    # model.BESS_pow_const = Constraint(rule=BESS_power)

    """
    Cyclic storage, causing problems
    """
    def BESS_circ(model):
        return model.BESS_st[0] == model.BESS_st[len(hlist)-1]
    model.BESS_circular = Constraint(rule=BESS_circ)

    for h in hlist[1:]:  # Energy balance in the storage systems
        previous_h = hlist[hlist.index(h)-1]
        model.storage_const.add(model.BESS_in[previous_h]*eta_BESS -
                                model.BESS_out[previous_h]*eta_BESS +
                                model.BESS_st[previous_h] == model.BESS_st[h])

    # for h in hlist[1:]: # Energy balance in the storage systems
    #     previous_h = hlist[hlist.index(h)-1]
    #     model.storage_const.add(model.BESS_in[previous_h ]*eta_BESS*model.BESS_charge[previous_h]-
    #                             model.BESS_out[previous_h]*eta_BESS*model.BESS_discharge[previous_h]+
    #                         model.BESS_st[previous_h ]==model.BESS_st[h])

    def BESS_c1(model, h):
        return (model.BESS_charge[h]+model.BESS_discharge[h]+model.BESS_idle[h] == 1)
    model.BESS_const = Constraint(hlist, rule=BESS_c1)

    def EZ_c1(model, h):
        return (model.EZ_E_in[h]*eta_EZ == model.EZ_E_out[h])
    model.EZ_const = Constraint(hlist, rule=EZ_c1)

    # def ObjRule(model):
    #     return model.EZ_size+model.BESS_cap
    # model.obj2 = pyo.Objective(rule=ObjRule)

    def ObjRule(model):
        return sum(model.EZ_E_in[h] for h in hlist)
    model.obj2 = pyo.Objective(rule=ObjRule, sense=maximize)

    # def ObjRule(model):
    #     return  (model.BESS_cap+model.BESS_power+1000*model.PV_size)
    # model.obj2 = pyo.Objective(rule=ObjRule, sense=minimize)

    return model


# %%


def df_summary(d, saving_fold):
    """
    Perform a summary analysis of the optimization results.

    Parameters:
        d (dict): A dictionary containing the data for each region.
        saving_fold (str): The path to the folder where the results will be saved.

    Returns:
        DataFrame: A DataFrame containing the summary results for each region.

    """

    eta_BESS = 0.9

    df_summary = pd.DataFrame(
        columns=["PV size (W)", "EZ size (W)", "BESS Cap [Wh]", "BESS Power [W]", "minimum lf", "Produced H2 (kg)"])

    for j in d.keys():
        de = d[j]
        lf = 0.95
        hlist = [i for i in range(len(de))]

        # instance = model.create_instance()
        opt = SolverFactory('gurobi')
        model = pyomo_model(lf, hlist, de)
        results = opt.solve(model)
        model.solutions.store_to(results)



        for step in np.arange(0.1, 1.0, 0.05)[::-1]:
            opt = pyo.SolverFactory('gurobi')

            model = pyomo_model(lf, hlist, de)
            results = opt.solve(model)

            if results.solver.termination_condition == TerminationCondition.optimal:
                model.solutions.store_to(results)

                print(f"Solution found for load factor minimum {step}:")
                break
            else:
                print(f"Solution NOT found for load factor {step}:")

        # %%
        df_res = pd.DataFrame(index=np.arange(0, len(hlist), 1))
        df_res["PV"] = [i*model.PV_size.extract_values()[None]
                        for i in (de[de.columns[1]]*1000).tolist()]
        df_res["Load"] = (de['load (kWh)']*1000).tolist()
        df_res["Electrolyser"] = model.EZ_E_in.extract_values()
        df_res["EZ_out"] = model.EZ_E_out.extract_values()

        df_res["$Battery_{in}$"] = model.BESS_in.extract_values()
        df_res["BESS_charge"] = model.BESS_charge.extract_values()
        df_res["$Battery_{out}$"] = model.BESS_out.extract_values()
        df_res["BESS_discharge"] = model.BESS_discharge.extract_values()
        df_res["$Battery_{storage}$"] = model.BESS_st.extract_values()
        df_res["BESS_idle state"] = model.BESS_idle.extract_values()

        # %%
        df_res.index = np.arange(1, len(hlist)+1, 1)
        df_res[['PV', 'Electrolyser', '$Battery_{in}$', '$Battery_{out}$', '$Battery_{storage}$', "Load"]].plot(subplots=True,
                                                                                                                kind="bar", layout=(3, 2), edgecolor="k", legend=False,
                                                                                                                ylabel="$[Wh]$", figsize=(10, 8))
        plt.suptitle(
            f"{name.split('.')[0]} "+j)
        plt.savefig(saving_fold+f"{name.split('.')[0]}"+j+".png", dpi=300)

        print(f"objective value is {value(model.obj2)}")

        df_plot = pd.DataFrame(index=df_res.index)
        df_plot['PV'] = df_res["PV"]
        df_plot['Electrolyser'] = -df_res["Electrolyser"]

        df_plot['$Battery_{in}$'] = -df_res["$Battery_{in}$"]*eta_BESS
        df_plot['$Battery_{out}$'] = df_res["$Battery_{out}$"]*eta_BESS
        df_plot['Load'] = -df_res["Load"]


        df_plot.plot.bar(stacked=True, edgecolor="black")
        ax = plt.gca()
        ax.set_ylabel("[Wh]")
        plt.ylim(-1500, +1500)

        plt.grid(axis="y", linestyle='--', linewidth=0.5, color="k")
        plt.legend(fancybox=False, edgecolor="k")

        plt.title(
            f"{name.split('.')[0]} "+j+f"  \n {round(model.PV_size.extract_values()[None],2)*1000} W PV, Electrolyser {round(value(model.EZ_size))} W \n  Battery:{round(value(model.BESS_power))} W with {round(value(model.BESS_cap))} Wh ")
        df_summary.loc[j] = [round(model.PV_size.extract_values()[None], 2)*1000,
                             model.EZ_size.extract_values()[None],
                             model.BESS_cap.extract_values()[None],
                             model.BESS_power.extract_values()[None], step, df_res["Electrolyser"].sum()/(1000*33.33)]
        plt.savefig(
            saving_fold+f"{name.split('.')[0]}"+j+"_balancing.png", dpi=300)

    df_summary["minimum load [kW]"] = df_summary.apply(
        lambda x: x["EZ size (W)"]*x["minimum lf"], axis=1)
    return df_summary

# %%

