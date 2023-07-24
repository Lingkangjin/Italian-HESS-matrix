# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:48:46 2023

@author: Lingkang Jin

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

class HESS():
    
    def __init__(self,province):
        self.province=province
        
        
        
    def load(self,):    
    df_c=pd.read_csv("consumption3.csv")
    # d=df_c[df_c["Provincia"]=="Ancona"]
    d=df_c[df_c["Provincia"]==self.province]

    d[d["Tipo Mercato"]== "Mercato Libero"]
    mesi=d["Anno Mese"].unique()[:-1]
    d[d["Anno Mese"]==d["Anno Mese"].unique()[0]]["Working Day"].unique()    


    #%%

    # Get national holidays in Italy for 2021
    it_holidays = holidays.IT(years=2021)

    # Create a list of Python datetime objects
    holiday_dates = [date for date in it_holidays.keys()]

    # Sort the list of dates
    holiday_dates.sort()

    # Print the list of holiday dates
    print(holiday_dates)


    #%%
    date_str = '01/01/2022'
    start = pd.to_datetime(date_str) - pd.Timedelta(days=365)
    hourly_periods = 8760
    drange = pd.date_range(start, periods=hourly_periods, freq='H')

    #%%

    mese=mesi[0]

    cond_residenza=(d["Residenza"]=='Tutti')
    cond_mercato=(d["Tipo Mercato"]=="Tutti")

    residenza="Tutti"
    de=d[ cond_residenza  & cond_mercato &(d["Anno Mese"]==mese) &(d["Working Day"]=="Domenica")]

    # drange[0].weekday()


    de['Prelive medio Orario Regionale (kWh)'] = de['Prelive medio Orario Regionale (kWh)'].str.replace(',', '.').astype(float).astype(float)

    #%%
    de['Prelive medio Orario Regionale (kWh)'].plot()
    #%%

    start_date = '2021-01-01'
    end_date = '2021-12-31'

    days = pd.date_range(start=start_date, end=end_date, freq='D').tolist()

    #%%
    Power=[]
    giorni=[]
    for i in days:
        if i.date() in holiday_dates: # check holidays  and substitute as sunday
            Power.extend(d[ cond_residenza  & cond_mercato &(d["Anno Mese"]==mesi[i.month-1]) &(d["Working Day"]=="Domenica")]["Prelive medio Orario Regionale (kWh)"].str.replace(',', '.').astype(float))
            giorni.extend(["Holiday"]*24)
        elif i.weekday()==6: #sunday
            Power.extend(d[ cond_residenza  & cond_mercato &(d["Anno Mese"]==mesi[i.month-1]) &(d["Working Day"]=="Domenica")]["Prelive medio Orario Regionale (kWh)"].str.replace(',', '.').astype(float))
            giorni.extend(["Sunday"]*24)
        elif i.weekday()==5: #saturday
            Power.extend(d[ cond_residenza  & cond_mercato &(d["Anno Mese"]==mesi[i.month-1]) &(d["Working Day"]=="Sabato")]["Prelive medio Orario Regionale (kWh)"].str.replace(',', '.').astype(float))
            giorni.extend(["Saturday"]*24)
        else:
            Power.extend(d[ cond_residenza  & cond_mercato &(d["Anno Mese"]==mesi[i.month-1]) &(d["Working Day"]=="Giorno feriale")]["Prelive medio Orario Regionale (kWh)"].str.replace(',', '.').astype(float))
            giorni.extend(["Working day"]*24)

     #%%
    df_load=pd.DataFrame(index=drange)
    df_load["load (kWh)"]=Power
    df_load["Days"]=giorni
        
    