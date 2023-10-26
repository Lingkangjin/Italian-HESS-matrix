# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:15:05 2023

@author: utente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import datetime

from pyomo.environ import *
from pyomo import environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HESS_reg_class import *

from tqdm import tqdm
import os

# %%

plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.constrained_layout.use"] = True

# %%

regions = ['Abruzzo', 'Basilicata', 'Calabria', 'Campania', 'Emilia-Romagna',
           'Friuli-Venezia Giulia', 'Lazio', 'Liguria', 'Lombardia', 'Marche',
           'Molise', 'Piemonte', 'Puglia', 'Sardegna', 'Sicilia', 'Toscana',
           'Trentino-Alto Adige', 'Umbria', "Valle d'Aosta", 'Veneto']
# %%
# df_load = pd.read_pickle("df_load.pkl")
# load = (df_load.iloc[:, 0]*1000).tolist()


# df_load = HESS("Marche").load()
# load = (df_load.iloc[:, 0]*1000).tolist()


# %% reading the entire year data
# df=pd.read_csv("PV_production.csv",index_col=0)
# df.dropna(axis=0,inplace=True)

# date_str = '31/12/2020'
# start = pd.to_datetime(date_str) - pd.Timedelta(days=365)
# hourly_periods = 8784
# drange = pd.date_range(start, periods=hourly_periods, freq='H')

# df.index=drange

# PV_s=[]
# import datetime
# a=datetime.datetime(2020,2,29)

# for i in df.index:
#     if i.date()==a.date():
#         pass
#     else:
#         PV_s.append(df.loc[i].P)


# PV_gen=PV_s

# %% reading from clusters


# %%

# model creation
# hlist=[i for i in range(8760)] #hours
def pyomo_model(lf, hlist, PV_gen):

    # parameters
    C_rate_bess_max = 1
    BESS_scale = 5  # maximum scale of power compared to PV size
    # hlist = [i for i in range(days*24)]

    eta_EZ = 0.7
    eta_BESS = 0.9

    initial_soc = 0.2
    SOC_min = 0.2
    M = 1000  # Big M value method to transfer the

    # hlist = [i for i in range(days*24)]  # hours

    model = ConcreteModel()  # Pyomo concrete model

    # Model Dispatch Decisions
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
        return (PV_gen[h] +
                model.BESS_out[h]*eta_BESS*(1 - model.BESS_charge[h]) -
                model.BESS_in[h]*eta_BESS*model.BESS_charge[h] -
                model.EZ_E_in[h]
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

    def BESS_power(model):
        return model.BESS_power <= PV_size*BESS_scale
    model.BESS_pow_const = Constraint(rule=BESS_power)

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
        return sum(model.EZ_E_in[h] for h in hlist)-(model.BESS_cap+model.BESS_power)
    model.obj2 = pyo.Objective(rule=ObjRule, sense=maximize)

    return model


# %%
PV_size = 1000  # 1kW
# df = pd.read_csv("k-means.csv", index_col=0)


# Specify the directory you want to list subfolders for
directory_path = 'PV data\\Clusters\\'

# Get a list of subfolders in the specified directory
subfolders = [f for f in os.listdir(directory_path) if os.path.isdir(
    os.path.join(directory_path, f))]

# Now, 'subfolders' contains a list of subfolder names within the specified directory
print(subfolders)


df = pd.read_csv("PV data\Clusters\Abruzzo\Abruzzo_clusters.csv", index_col=0)


# a = df.iloc[:, 0].tolist()

# for j in df.columns[1:]:
#     a.extend(df[j])

# a = [i if i > 5 else 0 for i in a]

def df_summary(df):
    df_summary = pd.DataFrame(
        columns=["EZ size (W)", "BESS Cap [Wh]", "BESS Power [W]", "minimum lf"])

    for o, j in tqdm(enumerate(df.columns)):
        PV_gen2 = []

        for k in df.iloc[:, o].tolist():
            if k < 5/PV_size:
                PV_gen2.append(0)
            else:
                PV_gen2.append(k*PV_size)
        PV_gen = PV_gen2

        # %% tunable parameters

        lf = 0.80
        hlist = [i for i in range(len(PV_gen))]

        # %% normal usage
        opt = SolverFactory('gurobi')
        model = pyomo_model(lf, hlist, PV_gen)
        results = opt.solve(pyomo_model(lf, hlist, PV_gen))
        model.solutions.store_to(results)
        # %% Design outputs

        # %% looping
        for step in np.arange(0.1, 1.0, 0.05)[::-1]:
            opt = pyo.SolverFactory('gurobi')

            model = pyomo_model(step, hlist, PV_gen)
            results = opt.solve(model)

            if results.solver.termination_condition == TerminationCondition.optimal:
                model.solutions.store_to(results)

                print(f"Solution found for load factor minimum {step}:")
                break
            else:
                print(f"Solution NOT found for load factor {step}:")

        # %% use value instead of extract

        # value(model.EZ_E_in)
        df_res = pd.DataFrame(index=np.arange(0, len(hlist), 1))
        df_res["PV"] = PV_gen[:len(hlist)]
        # df_res["EZ_on"]=model.EZ_on.extract_values()
        df_res["EZ_in"] = model.EZ_E_in.extract_values()
        df_res["EZ_out"] = model.EZ_E_out.extract_values()

        df_res["BESS_in"] = model.BESS_in.extract_values()
        df_res["BESS_charge"] = model.BESS_charge.extract_values()
        df_res["BESS_out"] = model.BESS_out.extract_values()
        df_res["BESS_discharge"] = model.BESS_discharge.extract_values()
        df_res["BESS_storage"] = model.BESS_st.extract_values()
        df_res["BESS_idle state"] = model.BESS_idle.extract_values()

        # %%

        # no load--> no 'EZ_out'
        # df_res.index = np.arange(1, len(hlist)+1, 1)
        # df_res[['PV', 'EZ_in', 'BESS_in', 'BESS_out', 'BESS_storage']].plot(subplots=True,
        #                                                                     kind="bar", layout=(3, 2), edgecolor="k", legend=False,
        #                                                                     ylabel="$[W]$")

        # print(f"objective value is {value(model.obj2)}")
        # # print(f"EZ size is {value(model.EZ_size)}")

        # df_plot = pd.DataFrame(index=df_res.index)
        # df_plot['PV'] = df_res["PV"]
        # df_plot['EZ_in'] = -df_res["EZ_in"]

        # df_plot['BESS_in'] = -df_res["BESS_in"]
        # # df_plot['EZ_out']=df_res["EZ_out"]
        # df_plot['BESS_out'] = df_res["BESS_out"]

        # # color_dict = {'PV': 'orange', 'EZ_in': 'green', 'Column3': 'green'}

        # df_plot.plot.bar(stacked=True, edgecolor="black")
        # ax = plt.gca()
        # ax.set_ylabel("[W]")
        # plt.grid(axis="y", linestyle='--', linewidth=0.5, color="k")
        # plt.legend(fancybox=False, edgecolor="k")
        # # (df_res["load"]*0.1).plot(ax=ax)

        # # plt.xticks(rotation=0)
        # plt.title(
        #     f"{j} EZ {round(value(model.EZ_size))} W \n  BESS:{round(value(model.BESS_power))} W with {round(value(model.BESS_cap))} Wh ")

        # plt.savefig("Scheduling.png", dpi=300)

        # %% plot disabled
        # df_plot["lf EZ"] = round(100*(df_res["EZ_in"]/model.EZ_size.value))

        # fig, ax = plt.subplots()

        # bars = ax.bar(df_plot["lf EZ"].index, df_plot["lf EZ"],
        #               edgecolor="k", label="Load factor", alpha=0.7)
        # for bar in bars:
        #     if bar.get_height() == 100:
        #         bar.set_hatch('X')
        #         bar.set_facecolor('#ff7f0e')

        # ax.bar_label(bars)
        # plt.xticks(np.arange(4, 25, 4))

        # plt.ylim(i*100-10, 100+5)

        # plt.ylabel("EZ load factor [%]")
        # plt.xlabel("Hours")
        # plt.title(f"{j}")

        # plt.savefig("load factor.png", dpi=300)

    # %% grouping values
        df_summary.loc[j] = [model.EZ_size.extract_values()[None],
                             model.BESS_cap.extract_values()[None],
                             model.BESS_power.extract_values()[None], step]

    df_summary["minimum load [kW]"] = df_summary.apply(
        lambda x: x["EZ size (W)"]*x["minimum lf"], axis=1)
    return df_summary

# df_summary[df_summary['EZ size (W)']==df_summary['EZ size (W)'].min()]


# %%

df_all = pd.DataFrame(columns=[['EZ size (W)', 'BESS Cap [Wh]', 'BESS Power [W]', 'minimum lf',
                                'minimum load [kW]']])


# Specify the directory you want to list subfolders for
directory_path = 'PV data\\Clusters\\'

# Get a list of subfolders in the specified directory
subfolders = [f for f in os.listdir(directory_path) if os.path.isdir(
    os.path.join(directory_path, f))]

# # Now, 'subfolders' contains a list of subfolder names within the specified directory
# print(subfolders)

for i in tqdm(subfolders):
    df = pd.read_csv(f"PV data\Clusters\{i}\{i}_clusters.csv", index_col=0)
    d = df_summary(df)
    df_all.loc[f"{i}"] = d[d['EZ size (W)'] ==
                           d['EZ size (W)'].min()].iloc[0, :].tolist()

# df_all.to_csv("Results\\No load\\summar.csv")
# %%
