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

# %%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.constrained_layout.use"] = True

# %%
df_load = pd.read_pickle("df_load.pkl")
load = (df_load.iloc[:, 0]*1000).tolist()


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
PV_size = 1000  # 1kW
df = pd.read_csv("k-means.csv", index_col=0)

PV_gen2 = []

for i in df.iloc[:, 0].tolist():
    if i < 5:
        PV_gen2.append(0)
    else:
        PV_gen2.append(i)
PV_gen = PV_gen2

# %% tunable parameters
days = 1


lf = 0.16
C_rate_bess_max = 1
BESS_scale = 2  # maximum scale of power compared to PV size
hlist = [i for i in range(days*24)]
# %%
# parameters
eta_EZ = 0.7
eta_BESS = 0.9

initial_soc = 0.2
SOC_min = 0.2
M = 1000  # Big M value method to transfer the


# %%

# model creation
# hlist=[i for i in range(8760)] #hours
def pyomo_model(lf):
    hlist = [i for i in range(days*24)]  # hours

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
                model.BESS_in[h]*model.BESS_charge[h] -
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
    # def BESS_circ(model):
    #     return  model.BESS_st[0]==model.BESS_st[len(hlist)-1]
    # model.BESS_circular = Constraint(rule=BESS_circ)

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

    return model


# %% normal usage
opt = pyo.SolverFactory('gurobi')
model = pyomo_model(lf)
results = opt.solve(pyomo_model(lf))
model.solutions.store_to(results)
# %% looping
for i in np.arange(0.1, 1.0, 0.05)[::-1]:
    opt = pyo.SolverFactory('gurobi')

    model = pyomo_model(i)
    results = opt.solve(model)

    if results.solver.termination_condition == TerminationCondition.optimal:
        model.solutions.store_to(results)

        print(f"Solution found for load factor minimum {i}:")
        break
    else:
        print(f"Solution NOT found for load factor {i}:")


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
if days <= 7:

    # df_res.plot(subplots=True,kind="bar",layout=(5,2),edgecolor="k",legend=False)

    # no load--> no 'EZ_out'
    df_res.index = np.arange(1, 24+1, 1)
    df_res[['PV', 'EZ_in', 'BESS_in', 'BESS_out', 'BESS_storage']].plot(subplots=True,
                                                                        kind="bar", layout=(3, 2), edgecolor="k", legend=False,
                                                                        ylabel="$[W]$")

    # fig, axes=plt.subplots(nrows=3,ncols=2,constrained_layout=True)

    # #PV
    # axes[0, 0].bar(df_res.index, df_res["PV"])
    # axes[0, 1].axis('off')

    print(f"objective value is {value(model.obj2)}")
    # print(f"EZ size is {value(model.EZ_size)}")

    # %%
    df_plot = pd.DataFrame(index=df_res.index)
    df_plot['PV'] = df_res["PV"]
    df_plot['EZ_in'] = -df_res["EZ_in"]

    df_plot['BESS_in'] = -df_res["BESS_in"]
    # df_plot['EZ_out']=df_res["EZ_out"]
    df_plot['BESS_out'] = df_res["BESS_out"]

    # color_dict = {'PV': 'orange', 'EZ_in': 'green', 'Column3': 'green'}

    df_plot.plot.bar(stacked=True, edgecolor="black")
    ax = plt.gca()
    plt.grid(axis="y", linestyle='--', linewidth=0.5, color="k")
    plt.legend(fancybox=False, edgecolor="k")
    # (df_res["load"]*0.1).plot(ax=ax)

    # plt.xticks(rotation=0)
    plt.title(
        f"EZ {round(value(model.EZ_size))} W \n  BESS:{round(value(model.BESS_power))} W with {round(value(model.BESS_cap))} Wh ")
else:
    df_res.plot(subplots=True)


# %%
df_plot["lf EZ"] = round(100*(df_res["EZ_in"]/model.EZ_size.value))

fig, ax = plt.subplots()


bars = ax.bar(df_plot["lf EZ"].index, df_plot["lf EZ"],
              edgecolor="k", label="Load factor", alpha=0.9)
for bar in bars:
    if bar.get_height() == 100:
        bar.set_hatch('X')
        bar.set_facecolor('#ff7f0e')

ax.bar_label(bars)
plt.xticks(np.arange(4, 25, 4))


plt.ylim(i*100-10, 100+5)

plt.ylabel("EZ load factor [%]")
plt.xlabel("Hours")


# %%

# nature_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']


# from pylab import *
# # plt.rcParams['image.cmap'] = 'gray'

# # cmap = cm.get_cmap('tab20', 24)    # PiYG

# # # Get the default color map
# # cmap = plt.cm.get_cmap(24)

# cmap=plt.cm.get_cmap()

# HEX_codes=[]
# for i in range(cmap.N):
#     rgba = cmap(i)
#     # rgb2hex accepts rgb or rgba
#     HEX_codes.append(matplotlib.colors.rgb2hex(rgba))
#     print(matplotlib.colors.rgb2hex(rgba))
