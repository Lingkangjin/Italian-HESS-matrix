# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:48:46 2023

@author: Lingkang Jin

"""
import pandas as pd
import holidays
import os

# %%


class HESS():
    """
    HESS Class

    This class represents the HESS (Hybrid Energy Storage System) model. It is used to load data for a specific regione and generate a dataframe with load information.

    Attributes:
        regione (str): The name of the regione for which the data is loaded.

    Methods:
        load(): Loads the data for the specified regione and generates a dataframe with load information.

    """

    def __init__(self, regione):
        self.regione = regione

    def load(self,):
        file_path= os.path.join(os.getcwd(), 'Input_data', 'reg_2022_cons.csv')
        df_c = pd.read_csv(file_path)

        d = df_c[df_c["Regione"] == self.regione]
        mesi = d["Anno Mese"].unique()[:-1]

        # Get national holidays in Italy for 2022
        it_holidays = holidays.IT(years=2022)

        # Create a list of Python datetime objects
        holiday_dates = [date for date in it_holidays.keys()]

        # Sort the list of dates
        holiday_dates.sort()

        # Print the list of holiday dates
        print(holiday_dates)

        # %% initialize the time index
        date_str = '01/01/2023'
        start = pd.to_datetime(date_str) - pd.Timedelta(days=365)
        hourly_periods = 8760
        drange = pd.date_range(start, periods=hourly_periods, freq='h')

        # %% filtering out the results


        cond_residenza = (d["Residenza"] == 'Tutti')
        cond_mercato = (d["Tipo Mercato"] == "Tutti")






        # %%

        start_date = '2022-01-01'
        end_date = '2022-12-31'

        days = pd.date_range(start=start_date, end=end_date, freq='D').tolist()

        # %% creating the power profile
        Power = []
        giorni = []
        for i in days:
            if i.date() in holiday_dates:  # check holidays  and substitute as sunday
                Power.extend(d[cond_residenza & cond_mercato & (d["Anno Mese"] == mesi[i.month-1]) & (d["Working Day"]
                             == "DOM")]["Prelive medio Orario Regionale (kWh)"])
                giorni.extend(["Holiday"]*24)
            elif i.weekday() == 6:  # sunday
                Power.extend(d[cond_residenza & cond_mercato & (d["Anno Mese"] == mesi[i.month-1]) & (d["Working Day"]
                             == "DOM")]["Prelive medio Orario Regionale (kWh)"])
                giorni.extend(["Sunday"]*24)
            elif i.weekday() == 5:  # saturday
                Power.extend(d[cond_residenza & cond_mercato & (d["Anno Mese"] == mesi[i.month-1]) & (
                    d["Working Day"] == "SAB")]["Prelive medio Orario Regionale (kWh)"])
                giorni.extend(["Saturday"]*24)
            else:
                Power.extend(d[cond_residenza & cond_mercato & (d["Anno Mese"] == mesi[i.month-1]) & (d["Working Day"]
                             == "Giorno_feriale")]["Prelive medio Orario Regionale (kWh)"])
                giorni.extend(["Working day"]*24)

         # %% create the dataframe
        df_load = pd.DataFrame(index=drange)
        df_load["load (kWh)"] = Power
        df_load["Days"] = giorni

        return df_load

HESS('Abruzzo')