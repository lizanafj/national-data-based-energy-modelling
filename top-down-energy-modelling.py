# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:24:10 2021

@author: Jesus Lizana

Instructions: 
    All functions and figures are defined following the analysis workflow in the same script. 
    This helps for a better understanding of every step to support further modifications. 
    
    The approach is divided into five steps: 
    # (1) HEATING 
    # (2) DEMAND SCENARIOS
    # (3) GRID SCENARIO
    # (4) ELECTRICITY LOAD
    # (5) GHG EMISSIONS
    
    Data sources: 
    INPUT: 
        all .csv files required for the anaysys are provided as a example - Data for UK and SP of 2018
    OUTPUT:
        CSV table is generated and saved with a summary of results per region and scenario
    
    
List of versions: 
    Version 0.01 Raw code without cleaning
    Version 0.02 Reduction of data points for dissemination of example
    Version 0.03 Structuring and organisation - elimination of unuseful code/material  
    Version 0.04 First code for git-hub 
        

"""

#Python Modules required

import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt #figures
import numpy as np #to work with arrays
import seaborn as sns #figures statistical data visualisation
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc



#%% 

###############################################################################################################################

#SELEC CASE STUDY and DEFINE INPUT DATA (1/1) and (1/2)

###############################################################################################################################


#%% 

#THESE ARE THE TWO VARIABLES THAT REQUIRE MODIFICATIONS TO RUN THE CODE - CASE STUDY AND VALUE OF SCENARIO S0 (BASELINE)

#Select case study (input folder)
case_study = "ES"  #ES for SPAIN /or GB for the UK


#REFERENCE OUTPUT VALUES to compare (Scenario - S0)
# 10% GHG emissions for heating in the country under study, Mtonnes CO2eq
ghg1819t = 1.45  #values for 2018: 1.45 for SPAIN (ES) / 6.50 for UK (GB)


#%% 

# get current location of script
cwd = os.path.dirname(__file__) 

#folder of input data of selected case study
folder = "\data\INPUT_"+case_study #ES or GB

#folder for output data of selected case study
output_folder = "\data\OUTPUT_"+case_study #ES or GB


#%% 

#INPUTS 1/2
os.chdir(cwd+folder)

#IMPORT DATA - EUROSTAT
#########################################################
#ANNUAL HEATING DEMAND (SPACE HEATING AND HOT WATER)
heating = pd.read_csv(case_study+'_Heating_TJ.csv',sep=";",index_col = "datetime", parse_dates=True) 
water = pd.read_csv(case_study+'_Water_TJ.csv', sep=";",index_col = "datetime", parse_dates=True) 

#MONTHLY HDD
hdd =  pd.read_csv(case_study+'_HDD.csv', sep=";") #,index_col = "Monthly", parse_dates=True

#INPUT DATA FROM INTERNATIONAL ENERGY AGENCY 
#efficiency: 
solid=0.72
gas=0.91
oil=0.72
efficiency = [solid,gas,oil]

#IMPORT DATA - ELECTRICITY MAP
#########################################################
df_electricity = pd.read_csv(case_study+'_ElectricityMap_2018-2019.csv', index_col = "datetime", parse_dates=True,usecols= ["datetime","carbon_intensity_avg","total_consumption_avg","power_production_wind_avg"]) 

marginal = pd.read_csv(case_study+'_marginal_emissions.csv', index_col = "datetime", parse_dates=True) 
marginal = marginal.truncate(before='01-01-2018', after='01-01-2020')
marginal.index = pd.to_datetime(marginal.index).tz_localize('Etc/UCT')
df_electricity = pd.concat([df_electricity,marginal],axis=1)


#IMPORT DATA - WHEN2HEAT 
#seasonal COP profile
seasonalCOP = pd.read_csv(case_study+'_monthly_COP_values.csv', index_col = "utc_timestamp", parse_dates=True) 
#eliminate first row: 
seasonalCOP = seasonalCOP.iloc[1: , :]

#Heating profile 
heatingProfile = pd.read_csv(case_study+'_monthly_residential_space_heating_demand_profiles.csv', index_col = "Date", parse_dates=True)
waterProfile = pd.read_csv(case_study+'_monthly_residential_water_heating_demand_profiles.csv', index_col = "Date", parse_dates=True)


#%%

#INPUTS 2/2

#OTHER INPUT VALUES: 

#factors
GRIDfactor = 1.05 

#Heating target - fraction of heating 
fraction= 0.1


#%%


###############################################################################################################################

# (1) HEATING DEMAND

###############################################################################################################################



#%% 


#FUNCTION 1 - CALCULATION OF MONTHLY HEATING DEMAND OF SPACE HEATING AND HOT WATER - TARGET 10% - BASELINE 

def DEMAND(heating,water,efficiency,hdd,fraction,heatingProfile,waterProfile):
    
    heating_demand = heating*efficiency
    water_demand = water*efficiency

    #demand anual de TJ a GWh /3.6
    heating_demand_annual = heating_demand.sum(axis=1).to_frame().rename(columns={0: "heating"})/3.6 
    water_demand_annual = water_demand.sum(axis=1).to_frame().rename(columns={0: "water"})/3.6
    
    heating_demand_annual["year"] = heating_demand_annual.index.year

    heating_demand_annual = heating_demand_annual.reset_index()
    water_demand_annual = water_demand_annual.reset_index()
    
    heating_demand = hdd.copy()
    
    heating_demand["Monthly"] = heating_demand["Monthly"].apply(pd.to_datetime)
    heating_demand["year"] = heating_demand["Monthly"].dt.year
    heating_demand["month"] = heating_demand["Monthly"].dt.month
    
    for i in range(0,len(heating_demand)):
        for m in range(0,len(heating_demand_annual)):
            if heating_demand.loc[i,"year"] == heating_demand_annual.loc[m,"year"]:
                heating_demand.loc[i,"heating(Annual-GWh)"] = heating_demand_annual.loc[m,"heating"]
                heating_demand.loc[i,"hotwater(Annual-GWh)"] = water_demand_annual.loc[m,"water"]
                i=i+1
            else:
                m=m+1
    
    total_heating_demand = heating_demand.groupby("year")['M,NR,HDD,ES'].sum().to_frame().reset_index()
    print(total_heating_demand)
    
    for i in range(0,len(heating_demand)):
        for m in range(0,len(total_heating_demand)):
            if heating_demand.loc[i,"year"] == total_heating_demand.loc[m,"year"]:
                heating_demand.loc[i,"monthly-heating(MWh)"] = fraction*(heating_demand.loc[i,"M,NR,HDD,ES"]/total_heating_demand.loc[m,"M,NR,HDD,ES"])*(heating_demand.loc[i,"heating(Annual-GWh)"]*1000)
                
                heating_demand.loc[i,"monthly-hotwater(MWh)"] = fraction*(1/12)*(heating_demand.loc[i,"hotwater(Annual-GWh)"]*1000)
                i=i+1
            else:
                m=m+1
    
    heating_demand["datetime"]=heating_demand["Monthly"]    
    heating_demand = heating_demand.set_index(["Monthly"])
    heating_demand.index = [d.strftime("%b %Y") for d in heating_demand.index]
       
    return heating_demand


#%%

#Apply function:
heating_demand = DEMAND(heating,water,efficiency,hdd,fraction,heatingProfile,waterProfile)

         
#%%

#PREPARATION OF NORMALISED DEMAND PATTERNS TO CALCULATE HOURLY DISTRIBUTION

#HEATING DEMAND PROFILE: 
#Space heating:
heatingProfile1 = heatingProfile.copy()
heatingProfile1["month"] = [d.strftime("%b") for d in heatingProfile1.index]
heatingProfile1 = heatingProfile1.reset_index().set_index("month").drop("Date",axis=1)
heatingProfile2 = heatingProfile1.T #transpose 
heatingProfile3 = heatingProfile2.reset_index()
heatingProfile3['index']= pd.to_datetime(heatingProfile3['index'])
heatingProfile3 = heatingProfile3.set_index("index")
heatingProfile3["hour"] = [d.strftime("%H:%M:%S") for d in heatingProfile3.index]
heatingProfile3 = heatingProfile3.reset_index().set_index("hour").drop("index",axis=1) 

del([heatingProfile1,heatingProfile2])

#Hot woter: 
waterProfile1 = waterProfile.copy()
waterProfile1["month"] = [d.strftime("%b") for d in waterProfile1.index]
waterProfile1 = waterProfile1.reset_index().set_index("month").drop("Date",axis=1)
waterProfile2 = waterProfile1.T #transpose 
waterProfile3 = waterProfile2.reset_index()
waterProfile3['index']= pd.to_datetime(waterProfile3['index'])
waterProfile3 = waterProfile3.set_index("index")
waterProfile3["hour"] = [d.strftime("%H:%M:%S") for d in waterProfile3.index]
waterProfile3 = waterProfile3.reset_index().set_index("hour").drop("index",axis=1)
del([waterProfile1,waterProfile2])


#%%

#COP PROFILE: 
seasonalCOP1 = seasonalCOP.copy()
#seasonalCOP1["month"] = [d.strftime("%b") for d in seasonalCOP1.index]
seasonalCOP1 = seasonalCOP1.rename(columns={case_study+'_COP_ASHP_radiator': 'COP_ASHP_spaceheating',case_study+'_COP_ASHP_water': 'COP_ASHP_hotwater'})


#%%

#check heating average
heatingProfile_mean= pd.DataFrame()
heatingProfile_mean["mean"] = heatingProfile3.mean(axis=1)

waterProfile_mean= pd.DataFrame()
waterProfile_mean["mean"] = waterProfile3.mean(axis=1)


print(heatingProfile_mean.sum()) #resul should be 1
print(waterProfile_mean.sum()) #resul should be 1


#%%

#Figure. Plot normalised demand patterns:
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rc('font',family='Arial',size=10)

#FIGURE. Heating demand profile 
import matplotlib.dates as md
fig, axs = plt.subplots(2,1, sharex=False, sharey=False,figsize=(6, 7))

axs[0].plot(heatingProfile3)
axs[0].plot(heatingProfile_mean, color="k",linestyle="--",linewidth=2)
axs[0].legend(heatingProfile3,loc='upper right',fontsize=9, ncol=3)
axs[0].set_ylim(0, 0.1)
#axs[0].set_xlim(0, 24)
axs[0].set_ylabel('Normalised demand')
axs[0].set_title("Hourly space heting demand",loc="left")
axs[0].set_xticks(np.arange(0, 24, 6))
axs[0].margins(x=0.00)

axs[1].plot(waterProfile3)
axs[1].plot(waterProfile_mean, color="k",linestyle="--",linewidth=2)
axs[1].legend(waterProfile3,loc='upper right',fontsize=8, ncol=3)
axs[1].set_ylim(0, 0.1)
#axs[1].set_xlim(0,24)
axs[1].set_ylabel('Normalised demand')
axs[1].set_title("Hourly hot water demand",loc="left")
axs[1].set_xticks(np.arange(0, 24, 6))
axs[1].margins(x=0.00)

plt.subplots_adjust(left=None, bottom=None, right=None, top=0.92, wspace=None, hspace=0.3)

plt.show()


#%%

#FIGURE. Seasonal COP of HP 
import matplotlib.dates as mdates
labels = ['Jan', 'Jan','Jan','Jan','Jan','Jan','Jan','Jan','Jan','Jan','Jan','Jan']

fig, axs = plt.subplots(figsize=(6, 4))

axs.plot(seasonalCOP1)
axs.legend(seasonalCOP1,loc='upper right',fontsize=9, ncol=1)
axs.set_ylim(0, 6)
#axs.set_xlim(0, 12)
axs.set_ylabel('COP')
axs.set_title("Seasonal COP ",loc="left")
axs.set_xticks(seasonalCOP1.index)
axs.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axs.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

plt.show()


#%%

#FUNCTION 2 - CALCULATION OF HOURLY HEATING DEMAND USING NORMALISED DEMAND PATTERNS

#### Distribution of profiles: 

#Create dataframe with same datetime structure 
heating_demand_hourly = df_electricity.copy()

#time for indexing
heating_demand_hourly["month"] = [d.strftime("%b %Y") for d in heating_demand_hourly.index]
heating_demand_hourly["month_only"] = [d.strftime("%b") for d in heating_demand_hourly.index]
heating_demand_hourly["hour"] = [d.strftime("%H:%M:%S") for d in heating_demand_hourly.index]
heating_demand_hourly = heating_demand_hourly.drop(["carbon_intensity_avg","total_consumption_avg","power_production_wind_avg","marginal_carbon_intensity_avg"], axis=1)

#Time for indexing 
seasonalCOP1["month"] = [d.strftime("%b") for d in seasonalCOP1.index]

#%%

#Reset index for conditionals for loop
heating_demand = heating_demand.reset_index()
heating_demand_hourly = heating_demand_hourly.reset_index()
heatingProfile_mean = heatingProfile_mean.reset_index()
waterProfile_mean = waterProfile_mean.reset_index()
seasonalCOP1 = seasonalCOP1.reset_index()

#%%

#distribution of monthly heating demand
for i in range(0,len(heating_demand_hourly)):
    for m in range(0,len(heating_demand)):
        if heating_demand_hourly.loc[i,"month"] == heating_demand.loc[m,"index"]:
            heating_demand_hourly.loc[i,"heating_demand"] = heating_demand.loc[m,'monthly-heating(MWh)']
            heating_demand_hourly.loc[i,"hotwater_demand"] = heating_demand.loc[m,'monthly-hotwater(MWh)']

      
#%%      

#distribution of hourly heating profiles
for i in range(0,len(heating_demand_hourly)):            
    for m in range(0,len(heatingProfile_mean)):
        if heating_demand_hourly.loc[i,"hour"] == heatingProfile_mean.loc[m,"hour"]:
            heating_demand_hourly.loc[i,"heating_profile"] = heatingProfile_mean.loc[m,'mean']
            heating_demand_hourly.loc[i,"water_profile"] = waterProfile_mean.loc[m,'mean']


#%%

#distribution of seasonal COP
for i in range(0,len(heating_demand_hourly)):            
    for m in range(0,len(seasonalCOP1)):
        if heating_demand_hourly.loc[i,"month_only"] == seasonalCOP1.loc[m,"month"]:
            heating_demand_hourly.loc[i,"COP_ASHP_spaceheating"] = seasonalCOP1.loc[m,"COP_ASHP_spaceheating"]
            heating_demand_hourly.loc[i,"COP_ASHP_hotwater"] = seasonalCOP1.loc[m,"COP_ASHP_hotwater"]


#%%

#set datetime as index
heating_demand_hourly = heating_demand_hourly.set_index("datetime")

#Column - Days in month
heating_demand_hourly["days"] = heating_demand_hourly.index.daysinmonth

#Evaluation of hourly heating demand for space heating and hot water
heating_demand_hourly["hourly_spaceheating"] = (heating_demand_hourly["heating_demand"]/heating_demand_hourly["days"])*heating_demand_hourly["heating_profile"]
heating_demand_hourly["hourly_hotwater"] = (heating_demand_hourly["hotwater_demand"]/heating_demand_hourly["days"])*heating_demand_hourly["water_profile"]


#%%

#FIGURE. f - MONTHLY THERMAL HEATING DEMAND (SPACE HEATING AND HOT WATER) - BASELINE - FRACTION OF 10% OF ANNUAL NON-RENEWABLE FEC FOR HEATING SECTOR

heating_demand = heating_demand.set_index("index")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rc('font',family='Arial',size=10)

width = 0.9       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(10, 4))
p1 = plt.bar(heating_demand.index,heating_demand["monthly-hotwater(MWh)"]/1000, width, color="orange",label='Hot water')
p2 = plt.bar(heating_demand.index, heating_demand["monthly-heating(MWh)"]/1000, width, bottom=heating_demand["monthly-hotwater(MWh)"]/1000, color="red",label='Space heating')

plt.margins(x=0.01)
plt.xticks(rotation='vertical')
plt.ylabel('Gigawatt hours (GWh)')
plt.title('Monthly thermal heating demand of space heating and hot water')
ax.legend()
plt.yticks(np.arange(0, 5000, 1000))
#plt.legend((p1[0], p2[0]), ('PEC of Heating', 'PEC of Hot water'))

plt.show()



#%%

#FIGURE. EXAMPLE OF HOURLY THERMAL HEATING DEMAND - MW - BASELINE

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rc('font',family='Arial',size=10)

styles_a = ["orange", 'red']


heating_demand_hourly.truncate(before='2018-01-01', after='2018-01-15').plot.area(y=["hourly_hotwater","hourly_spaceheating"],
                       figsize=(12,3), ylim=[0,10000],style= styles_a,linewidth=0,legend=True)
plt.title("Hourly thermal heating demand")
plt.xlabel("datetime(Days)")
plt.ylabel("MW")

ax = df_electricity["carbon_intensity_avg"].truncate(before='2018-01-01', after='2018-01-15').plot(secondary_y=True, color='k',linestyle="--")
ax.set_ylabel("gCO2eq/kWh")
ax.set_ylim(0,600)

plt.show()




#Here we have calculated the hourly thermal heating demand profile used as baseline. 


#%%




###############################################################################################################################

# (2.1) DEMAND SCENARIOS + (4) ELECTRICITY LOAD

###############################################################################################################################




#%%

#CALCULATION OF ADDITIONAL ELECTRICITY LOAD
#SCENARIO S1. BASELINE. Additional electricity load associated with this heating electrification volume
heating_demand_hourly["additional_load"] = ((heating_demand_hourly["hourly_hotwater"]/heating_demand_hourly["COP_ASHP_hotwater"])+(heating_demand_hourly["hourly_spaceheating"]/heating_demand_hourly["COP_ASHP_spaceheating"]))*GRIDfactor
print("total: ", heating_demand_hourly["additional_load"].sum())

#SCENARIO S2. BASELINE + Efficiency 20%
heating_demand_hourly["additional_load"] = ((heating_demand_hourly["hourly_hotwater"]/heating_demand_hourly["COP_ASHP_hotwater"])+(heating_demand_hourly["hourly_spaceheating"]*0.8/heating_demand_hourly["COP_ASHP_spaceheating"]))*GRIDfactor
print("efficiency 20%: ", heating_demand_hourly["additional_load"].sum())



#Here we have calculated the additional wind production to match the additional electricity demand for heating.

#%%


###############################################################################################################################

# (3) GRID SCENARIO 

###############################################################################################################################



#%%


#WIND PROYECTION ACCORDING TO BASELINE SCENARIO S2: 
#-------------------------------------------------------------------
    
#Wind fraction for additional wind capacity
wind_fraction = 1*(heating_demand_hourly["additional_load"].sum()/df_electricity["power_production_wind_avg"].sum())

#Additional wind capacity - hourly 
df_electricity["additional_wind"] = df_electricity["power_production_wind_avg"]*wind_fraction   # additional wind = additional heating 

print("Additional wind capacity: ", df_electricity["additional_wind"].sum()) 

#save baseline
df_electricity["carbon_intensity_avg_base"] = df_electricity["carbon_intensity_avg"] #BACK UP OF BASELINE VALUE

#Calculation of updated carbon intesity with additional wind 
df_electricity["carbon_intensity_avg"] = (df_electricity["carbon_intensity_avg"] * df_electricity["total_consumption_avg"] - 490*df_electricity["additional_wind"])/(df_electricity["total_consumption_avg"]-df_electricity["additional_wind"])  #quitamos additional wind as gas
df_electricity["carbon_intensity_avg"] = (df_electricity["carbon_intensity_avg"]*(df_electricity["total_consumption_avg"]-df_electricity["additional_wind"]) + 11*df_electricity["additional_wind"])/ (df_electricity["total_consumption_avg"]) #añadimos wind

#%%
df_electricity = df_electricity.reset_index()

for i in range(0,len(df_electricity)):
    if df_electricity.loc[i,"carbon_intensity_avg"] < 0:
        df_electricity.loc[i,"carbon_intensity_avg"] = 0
    
df_electricity = df_electricity.set_index("datetime")



#%%

###############################################################################################################################

# (2.2) DEMAND SCENARIOS 

###############################################################################################################################


#%%

df_all = pd.concat([heating_demand_hourly,df_electricity],axis=1)


#%%

#FUNCTION - CALCULATION OF DEMAND RESPONSE SCENARIOS 

##Definition of input variables:
    
    #TES: Thermal energy storage capacity - Heat storage capacities are defined in relation to the time (in hours) in which the heating demand can be provided directly from the heat battery, without heat pump operation
    #losses: Energy consumption increase due to heat losses (%)
    
    #mode: pre-heating (1) or heat storage (2) - only mode 2 was used in this work
    
    #efficiency: Reduction of heating demand by improvements in the energy efficiency of buildings. Similar criteria than Scenario S2 was implemented.
    #maxPower: Maximum heating capacity constraint (cc, MW) 1 means power can not be increased in relation to baseline



def SCENARIOS(df_all,TES,losses,mode,efficciency,maxPower):
    
    #ON/OFF ACCORDING TO 50TH PERCENTILE EVERY X TIME
    df_Scenario = df_all[["heating_demand","hotwater_demand","days","heating_profile","water_profile","COP_ASHP_spaceheating","COP_ASHP_hotwater","hourly_spaceheating","hourly_hotwater",'carbon_intensity_avg']]

    #maximun heating capacity - similar to S1. baseline 
    df_Scenario["baseline_demand"] = df_Scenario["hourly_spaceheating"]*maxPower+df_Scenario["hourly_hotwater"]
    
    #starting point for scenario_demand
    df_Scenario["scenario_demand"] = (df_Scenario["hourly_spaceheating"]*efficciency+df_Scenario["hourly_hotwater"])*2
    
    power = 0
   
    if mode==2: #heat storage
    
        while df_Scenario["scenario_demand"].max()>df_Scenario["baseline_demand"].max():
        
            #Calculation on/off
            g1 = df_Scenario.resample(TES)['carbon_intensity_avg']
            df_Scenario[TES] = g1.transform('median') #percentil 
            df_Scenario['on/off'] = np.where(df_Scenario['carbon_intensity_avg'] < df_Scenario[TES], 1, power*df_Scenario['heating_profile']) #favorable 50th percentile
            
            ###SPACE HEATING
            #Correction factor to increase heating capacity in profile: 
            df_Scenario['S_spaceheating_demand']= df_Scenario['on/off'] #new profile
            df_Scenario['S_spaceheating_demand'] = (df_Scenario["heating_demand"]*efficciency/df_Scenario["days"])*df_Scenario["S_spaceheating_demand"] #demand
            
            factor_spaceheating1 = df_Scenario["hourly_spaceheating"].sum()*efficciency/df_Scenario['S_spaceheating_demand'].sum()
            
            #calculation of new heating demand profile including factor
            df_Scenario['S_spaceheating_demand'] = df_Scenario['on/off']*factor_spaceheating1 #new profile
            df_Scenario['S_spaceheating_demand'] = (df_Scenario["heating_demand"]*efficciency/df_Scenario["days"])*df_Scenario['S_spaceheating_demand'] #demand
            df_Scenario['S_spaceheating_demand'] = df_Scenario['S_spaceheating_demand']*losses #losses
            
            ###HOT WATER
            #Correction factor to increase heating capacity in profile: 
            df_Scenario['S_hotwater_demand']= df_Scenario['on/off'] #new profile
            df_Scenario['S_hotwater_demand'] = (df_Scenario["hotwater_demand"]/df_Scenario["days"])*df_Scenario["S_hotwater_demand"] #demand
            
            factor_spaceheating2 = df_Scenario["hourly_hotwater"].sum()/df_Scenario['S_hotwater_demand'].sum()
            
            #calculation of new heating demand profile including factor
            df_Scenario['S_hotwater_demand'] = df_Scenario['on/off']*factor_spaceheating2 #new profile
            df_Scenario['S_hotwater_demand'] = (df_Scenario["hotwater_demand"]/df_Scenario["days"])*df_Scenario['S_hotwater_demand'] #demand
            df_Scenario['S_hotwater_demand'] = df_Scenario['S_hotwater_demand']*losses #losses
            
            df_Scenario["scenario_demand"] = (df_Scenario['S_spaceheating_demand'] + df_Scenario['S_hotwater_demand'])
        
            power = power+0.01
            print(power, "HC_s :", df_Scenario["scenario_demand"].max(), "HC_baseline: ",df_Scenario["baseline_demand"].max(), factor_spaceheating1)

       
    df_Scenario_final = df_Scenario[["S_spaceheating_demand","S_hotwater_demand","scenario_demand","COP_ASHP_spaceheating","COP_ASHP_hotwater"]]        
            
    return df_Scenario_final


#%%

s1 = df_all[["hourly_spaceheating","hourly_hotwater","COP_ASHP_spaceheating","COP_ASHP_hotwater"]]
s1 = s1.rename(columns={'hourly_spaceheating': 'S_spaceheating_demand', 'hourly_hotwater': 'S_hotwater_demand'})
s1["scenario_demand"] = s1["S_spaceheating_demand"]+s1["S_hotwater_demand"]

s2 = df_all[["hourly_spaceheating","hourly_hotwater","COP_ASHP_spaceheating","COP_ASHP_hotwater"]]
s2 = s2.rename(columns={'hourly_spaceheating': 'S_spaceheating_demand', 'hourly_hotwater': 'S_hotwater_demand'})
s2["S_spaceheating_demand"] = s2["S_spaceheating_demand"]*0.80
s2["S_hotwater_demand"] = s2["S_hotwater_demand"]*1
s2["scenario_demand"] = s2["S_spaceheating_demand"]+s1["S_hotwater_demand"]

#%%

    
s3 = SCENARIOS(df_all,"2h",1.01,2,0.8,0.70)
s4 = SCENARIOS(df_all,"4h",1.015,2,0.8,0.75)
s5 = SCENARIOS(df_all,"6h",1.02,2,0.8,0.80)
s6 = SCENARIOS(df_all,"12h",1.04,2,0.8,1.00)
s7 = SCENARIOS(df_all,"24h",1.08,2,0.8,1.05)
s8 = SCENARIOS(df_all,"36h",1.12,2,0.8,1.05)
s9 = SCENARIOS(df_all,"48h",1.15,2,0.8,1.1)
s10 = SCENARIOS(df_all,"60h",1.18,2,0.8,1.1)


#%%

#Figure of heating demand - only check 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rc('font',family='Arial',size=10)

styles_a = ["orange", 'red']

#change this df to verify each 
s8.truncate(before='2018-01-01', after='2018-01-15').plot.area(y=["S_hotwater_demand","S_spaceheating_demand"],
                       figsize=(12,3), ylim=[0,30000],style= styles_a,linewidth=0,legend=True)
plt.title("Hourly thermal energy demand")
plt.xlabel("datetime(Days)")
plt.ylabel("Heating Capacity (MW)")


ax = df_all["carbon_intensity_avg"].truncate(before='2018-01-01', after='2018-01-15').plot(secondary_y=True, color='k',linestyle="--")
ax.set_ylabel("gCO2eq/kWh")
ax.set_ylim(0,600)

plt.show()



#%%

###############################################################################################################################

# (4) ELECTRICITY LOAD

###############################################################################################################################

#%%


List_df = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10]

for i in List_df:
    i["additional_load"] =  (i["S_spaceheating_demand"]/i["COP_ASHP_spaceheating"] +i["S_hotwater_demand"]/i["COP_ASHP_hotwater"])*GRIDfactor


#%%

###############################################################################################################################

# (5) GHG EMISSIONS

###############################################################################################################################
    
#%%

def GHG(s1aa,df_all):
    s1aa = pd.concat([s1aa,df_all[["carbon_intensity_avg","total_consumption_avg","marginal_carbon_intensity_avg"]]],axis=1)
    s1aa["updated_carbon_intensity"] = (s1aa["carbon_intensity_avg"]*s1aa["total_consumption_avg"] + s1aa["marginal_carbon_intensity_avg"]*s1aa["additional_load"])/ (s1aa["total_consumption_avg"]+s1aa["additional_load"])
    s1aa["Total_GHG"] = s1aa["additional_load"]*1000*s1aa["updated_carbon_intensity"]
    s1aa["Total_GHG_maginal"] = s1aa["additional_load"]*1000*s1aa["marginal_carbon_intensity_avg"]
    s1aa["Total_MW"]=s1aa["total_consumption_avg"]+s1aa["additional_load"]
    return s1aa


#%%
    
s1 = GHG(s1,df_all)
s2 = GHG(s2,df_all)
s3 = GHG(s3,df_all)
s4 = GHG(s4,df_all)
s5 = GHG(s5,df_all)
s6 = GHG(s6,df_all)
s7 = GHG(s7,df_all)
s8 = GHG(s8,df_all)
s9 = GHG(s9,df_all)
s10 = GHG(s10,df_all)


#%%


########################################################################################

# End of simulation

########################################################################################

            
#%%


#From here, all results are analysed: 
    








            
#%%

#FIGURE. DATA AND PLOT - Example of load shifting per DR scenario during 15 days from 01st January 2018

s1a =s1
s2a =s2
s3a =s3
s4a =s4
s5a =s5
s6a =s6
s7a =s7
s8a =s8
s9a =s9
s10a =s10

s1a= s1a.truncate(before='2018-01-01', after='2018-01-15')
s2a= s2a.truncate(before='2018-01-01', after='2018-01-15')
s3a= s3a.truncate(before='2018-01-01', after='2018-01-15')
s4a= s4a.truncate(before='2018-01-01', after='2018-01-15')
s5a= s5a.truncate(before='2018-01-01', after='2018-01-15')
s6a= s6a.truncate(before='2018-01-01', after='2018-01-15')
s7a= s7a.truncate(before='2018-01-01', after='2018-01-15')
s8a= s8a.truncate(before='2018-01-01', after='2018-01-15')
s9a= s9a.truncate(before='2018-01-01', after='2018-01-15')
s10a= s10a.truncate(before='2018-01-01', after='2018-01-15')

List_df_1 = [s1a,s2a,s3a,s4a,s5a,s6a,s7a,s8a,s9a,s10a]

for i in List_df_1:
    i["index"] = np.arange(len(i))
   
#%% 

#PLOT

List= ["s1. Baseline ",
       "s2. Baseline + 20% improvement in energy efficiency",
       "s3.Hourly DR = 20% energy efficiency + TES=2h",
       "s4.Hourly DR = 20% energy efficiency + TES=4h",
       "s5.Hourly DR = 20% energy efficiency + TES=6h",
       "s6.Hourly DR = 20% energy efficiency + TES=12h",
       "s7.Daily DR = 20% energy efficiency + TES=24h",
       "s8.Daily DR = 20% energy efficiency + TES=36h",
       "s9.Daily DR = 20% energy efficiency + TES=48h",
       "s10.Daily DR = 20% energy efficiency + TES=60h"]

width=1
color="red"
fig, axs = plt.subplots(10, sharex=True, sharey=True,figsize=(12, 14))
fig.suptitle('Load shifting strategies')
axs[0].bar(s1a["index"],s1a["additional_load"], width, color=color,label='PEC of Hot water')
ax1 = axs[0].twinx()
ax1.plot(s1a["index"],s1a["carbon_intensity_avg"], color='k',linestyle="--")
ax1.plot(s1a["index"],s1a["updated_carbon_intensity"], color='b',linestyle=":")
ax1.set_ylim(0, 500)
ax1.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[1].bar(s2a["index"],s2a["additional_load"], width, color=color,label='PEC of Hot water')
ax2 = axs[1].twinx()
ax2.plot(s10a["index"],s2a["carbon_intensity_avg"], color='k',linestyle="--")
ax2.plot(s2a["index"],s2a["updated_carbon_intensity"], color='b',linestyle=":")
ax2.set_ylim(0, 500)
ax2.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[2].bar(s3a["index"],s3a["additional_load"], width, color=color,label='PEC of Hot water')
ax3 = axs[2].twinx()
ax3.plot(s10a["index"],s3a["carbon_intensity_avg"], color='k',linestyle="--")
ax3.plot(s3a["index"],s3a["updated_carbon_intensity"], color='b',linestyle=":")
ax3.set_ylim(0, 500)
ax3.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[3].bar(s4a["index"],s4a["additional_load"], width, color=color,label='PEC of Hot water')
ax4 = axs[3].twinx()
ax4.plot(s10a["index"],s4a["carbon_intensity_avg"], color='k',linestyle="--")
ax4.plot(s4a["index"],s4a["updated_carbon_intensity"], color='b',linestyle=":")
ax4.set_ylim(0, 500)
ax4.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[4].bar(s5a["index"],s5a["additional_load"], width, color=color,label='PEC of Hot water')
ax5 = axs[4].twinx()
ax5.plot(s10a["index"],s5a["carbon_intensity_avg"], color='k',linestyle="--")
ax5.plot(s5a["index"],s5a["updated_carbon_intensity"], color='b',linestyle=":")
ax5.set_ylim(0, 500)
ax5.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[5].bar(s6a["index"],s6a["additional_load"], width, color=color,label='PEC of Hot water')
ax6 = axs[5].twinx()
ax6.plot(s10a["index"],s6a["carbon_intensity_avg"], color='k',linestyle="--")
ax6.plot(s6a["index"],s6a["updated_carbon_intensity"], color='b',linestyle=":")
ax6.set_ylim(0, 500)
ax6.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[6].bar(s7a["index"],s7a["additional_load"], width, color=color,label='PEC of Hot water')
ax7 = axs[6].twinx()
ax7.plot(s10a["index"],s7a["carbon_intensity_avg"], color='k',linestyle="--")
ax7.plot(s7a["index"],s7a["updated_carbon_intensity"], color='b',linestyle=":")
ax7.set_ylim(0, 500)
ax7.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[7].bar(s8a["index"],s8a["additional_load"], width, color=color,label='PEC of Hot water')
ax8 = axs[7].twinx()
ax8.plot(s10a["index"],s8a["carbon_intensity_avg"], color='k',linestyle="--")
ax8.plot(s8a["index"],s8a["updated_carbon_intensity"], color='b',linestyle=":")
ax8.set_ylim(0, 500)
ax8.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[8].bar(s9a["index"],s9a["additional_load"], width, color=color,label='PEC of Hot water')
ax9 = axs[8].twinx()
ax9.plot(s10a["index"],s9a["carbon_intensity_avg"], color='k',linestyle="--")
ax9.plot(s9a["index"],s9a["updated_carbon_intensity"], color='b',linestyle=":")
ax9.set_ylim(0, 500)
ax9.set_ylabel("gCO2eq/kWh",fontsize=8)

axs[9].bar(s10a["index"],s10a["additional_load"], width, color=color,label='PEC of Hot water')
ax10 = axs[9].twinx()
ax10.plot(s10a["index"],s10a["carbon_intensity_avg"], color='k',linestyle="--")
ax10.plot(s10a["index"],s10a["updated_carbon_intensity"], color='b',linestyle=":")
ax10.set_ylim(0, 500)
ax10.set_ylabel("gCO2eq/kWh",fontsize=8)

for i in range(10):
    #axs[i].set_title('Scenario '+str(i+1))
    axs[i].set_ylabel('MW')
    axs[i].set_ylim(0, 5000)
    #axs[i].grid(axis='x')
    axs[i].set_title(List[i],loc="left",fontsize=10)
    axs[i].margins(x=0.00)
    
    
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.4)
#plt.margins(x=0.00)

#plt.ylim(0, 3000)
#plt.yticks(np.arange(0, 10000, 5000))
plt.xticks(np.arange(0, len(s7a), 24))
plt.xlabel('Hours')

plt.show()

                
#%%


###########################################3###########################################3###########################################3

#FINAL SUMMARY TABLE: 

df_GHG = pd.DataFrame()

df_GHG.loc[0,"name"] = "GHG emissions MtonnesCO2eq"
df_GHG.loc[0,"S0.Fossil fuels"] = ghg1819t
df_GHG.loc[0,"S1.No TES"] = s1["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S2.No TES"] = s2["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S3.TES=2h"] = s3["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S4.TES=4h"] = s4["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S5.TES=6h"] = s5["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S6.TES=12h"] = s6["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S7.TES=24h"] = s7["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S8.TES=36h"] = s8["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S9.TES=48h"] = s9["Total_GHG"].sum()/1000000000000
df_GHG.loc[0,"S10.TES=60h"] = s10["Total_GHG"].sum()/1000000000000
df_GHG = df_GHG.set_index("name")

df_GHG_marg = pd.DataFrame()

df_GHG_marg.loc[0,"name"] = "Marginal GHG emissions MtonnesCO2eq"
df_GHG_marg.loc[0,"S0.Fossil fuels"] = ghg1819t
df_GHG_marg.loc[0,"S1.No TES"] = s1["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S2.No TES"] = s2["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S3.TES=2h"] = s3["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S4.TES=4h"] = s4["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S5.TES=6h"] = s5["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S6.TES=12h"] = s6["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S7.TES=24h"] = s7["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S8.TES=36h"] = s8["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S9.TES=48h"] = s9["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg.loc[0,"S10.TES=60h"] = s10["Total_GHG_maginal"].sum()/1000000000000
df_GHG_marg = df_GHG_marg.set_index("name")


df_GHG_p = 100*df_GHG/ghg1819t
df_GHG_p = df_GHG_p.round(3).astype(str) + "%"


#%%


#Maximum power demand of the additional heating load (MW)

maxLOAD = pd.DataFrame()

maxLOAD.loc[0,"name"] = "Maximum power demand of additional heating load (MW)"
maxLOAD.loc[0,"S0.Fossil fuels"] = np.nan
maxLOAD.loc[0,"S1.No TES"] = (s1["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S2.No TES"] = (s2["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S3.TES=2h"] = (s3["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S4.TES=4h"] = (s4["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S5.TES=6h"] = (s5["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S6.TES=12h"] = (s6["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S7.TES=24h"] = (s7["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S8.TES=36h"] = (s8["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S9.TES=48h"] = (s9["additional_load"].replace(0, np.NaN).max())
maxLOAD.loc[0,"S10.TES=60h"] = (s10["additional_load"].replace(0, np.NaN).max())
maxLOAD = maxLOAD.set_index("name")

#Maximum power demand of total consumption (MW)

maxTOTAL = pd.DataFrame()

maxTOTAL.loc[0,"name"] = "Maximum power demand of total consumption"
maxTOTAL.loc[0,"S0.Fossil fuels"] = np.nan
maxTOTAL.loc[0,"S1.No TES"] = (s1["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S2.No TES"] = (s2["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S3.TES=2h"] = (s3["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S4.TES=4h"] = (s4["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S5.TES=6h"] = (s5["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S6.TES=12h"] = (s6["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S7.TES=24h"] = (s7["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S8.TES=36h"] = (s8["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S9.TES=48h"] = (s9["Total_MW"].replace(0, np.NaN).max())
maxTOTAL.loc[0,"S10.TES=60h"] = (s10["Total_MW"].replace(0, np.NaN).max())
maxTOTAL = maxTOTAL.set_index("name")

#99% Percentile -  power demand of total consumption (MW)

percen = 0.99
#print(s1.Total_MW.quantile(percen))

PthmaxTOTAL = pd.DataFrame()

PthmaxTOTAL.loc[0,"name"] = "99th percentile of the power demand of total consumption"
PthmaxTOTAL.loc[0,"S0.Fossil fuels"] = np.nan
PthmaxTOTAL.loc[0,"S1.No TES"] = (s1["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S2.No TES"] = (s2["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S3.TES=2h"] = (s3["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S4.TES=4h"] = (s4["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S5.TES=6h"] = (s5["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S6.TES=12h"] = (s6["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S7.TES=24h"] = (s7["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S8.TES=36h"] = (s8["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S9.TES=48h"] = (s9["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL.loc[0,"S10.TES=60h"] = (s10["Total_MW"].replace(0, np.NaN).quantile(percen))
PthmaxTOTAL = PthmaxTOTAL.set_index("name")



increaseMW = pd.DataFrame()

increaseMW.loc[0,"name"] = "Increase in maximum power demand (99th percentile MW)"
increaseMW.loc[0,"S0.Fossil fuels"] = np.nan
increaseMW.loc[0,"S1.No TES"] = (s1["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S2.No TES"] = (s2["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S3.TES=2h"] = (s3["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S4.TES=4h"] = (s4["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S5.TES=6h"] = (s5["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S6.TES=12h"] = (s6["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S7.TES=24h"] = (s7["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S8.TES=36h"] = (s8["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S9.TES=48h"] = (s9["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW.loc[0,"S10.TES=60h"] = (s10["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))
increaseMW = increaseMW.set_index("name")


#Increase in maximum power demand (%)

increasePerc = pd.DataFrame()

increasePerc.loc[0,"name"] = "Increase in maximum power demand (99th percentile %)"
increasePerc.loc[0,"S0.Fossil fuels"] = np.nan
increasePerc.loc[0,"S1.No TES"] =  (s1["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S2.No TES"] = (s2["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S3.TES=2h"] = (s3["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S4.TES=4h"] = (s4["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S5.TES=6h"] = (s5["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S6.TES=12h"] = (s6["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S7.TES=24h"] = (s7["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S8.TES=36h"] = (s8["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S9.TES=48h"] = (s9["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc.loc[0,"S10.TES=60h"] = (s10["Total_MW"].replace(0, np.NaN).quantile(percen)-s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen))/s1["total_consumption_avg"].replace(0, np.NaN).quantile(percen)
increasePerc = increasePerc.set_index("name")
increasePerc = increasePerc*100
increasePerc = increasePerc.round(2).astype(str) + "%"

#Additional electricity load of heating GWh

HeatingGWh = pd.DataFrame()

HeatingGWh.loc[0,"name"] = "Additional electricity load of heating GWh"
HeatingGWh.loc[0,"S1.No TES"] = s1["additional_load"].sum()/1000
HeatingGWh.loc[0,"S2.No TES"] = s2["additional_load"].sum()/1000
HeatingGWh.loc[0,"S3.TES=2h"] = s3["additional_load"].sum()/1000
HeatingGWh.loc[0,"S4.TES=4h"] = s4["additional_load"].sum()/1000
HeatingGWh.loc[0,"S5.TES=6h"] = s5["additional_load"].sum()/1000
HeatingGWh.loc[0,"S6.TES=12h"] = s6["additional_load"].sum()/1000
HeatingGWh.loc[0,"S7.TES=24h"] = s7["additional_load"].sum()/1000
HeatingGWh.loc[0,"S8.TES=36h"] = s8["additional_load"].sum()/1000
HeatingGWh.loc[0,"S9.TES=48h"] = s9["additional_load"].sum()/1000
HeatingGWh.loc[0,"S10.TES=60h"] = s10["additional_load"].sum()/1000
HeatingGWh = HeatingGWh.set_index("name")

#Electricity consumption (without heating) GWh

BaselineGWh = pd.DataFrame()

BaselineGWh.loc[0,"name"] = "Electricity consumption (without heating) GWh"
BaselineGWh.loc[0,"S1.No TES"] = s1["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S2.No TES"] = s2["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S3.TES=2h"] = s3["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S4.TES=4h"] = s4["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S5.TES=6h"] = s5["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S6.TES=12h"] = s6["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S7.TES=24h"] = s7["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S8.TES=36h"] = s8["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S9.TES=48h"] = s9["total_consumption_avg"].sum()/1000
BaselineGWh.loc[0,"S10.TES=60h"] = s10["total_consumption_avg"].sum()/1000
BaselineGWh = BaselineGWh.set_index("name")


#%%

#Table generation and saved as csv

SummaryGHG= pd.DataFrame()
SummaryGHG = pd.concat([df_GHG,df_GHG_p,df_GHG_marg,maxLOAD,maxTOTAL,PthmaxTOTAL,increaseMW,increasePerc,HeatingGWh,BaselineGWh],axis=0)

os.chdir(cwd+output_folder)
SummaryGHG.to_csv(r"SummaryGHG_GB.csv", index = True , header=True)


#%% 

#FIGURE. GHG emissions per scenario

#FUNCTION TO ADD LABELS IN BARS - only see

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

# seleccionar ghg or marginal
ax = df_GHG.transpose().plot(kind='bar', figsize=(10, 4), title='Amount Frequency',
             xlabel=None, ylabel='Mtones CO2eq', legend=False,width = 0.8,
             color="#8C9CB5",  edgecolor='#636B84')

plt.title('Impact of heat storage through smart DR in heating electrification')
ax.set_ylim(0, 15)
ax.margins(y=0.1)
#ax.bar_label(values1)

# Call the function above. All the magic happens there.
add_value_labels(ax)

fig.tight_layout()

plt.show()

            
#%% 

##FIGURE. Power duration curves 2018 and 2019

#################################################################################################################################

#Data 2018
#truncate(before='2018-01-01', after='2019-01-01').
s0_2018 = df_electricity.truncate(before='2018-01-01', after='2019-01-01')
s1_2018 = s1.truncate(before='2018-01-01', after='2019-01-01')
s2_2018 = s2.truncate(before='2018-01-01', after='2019-01-01')
s3_2018 = s3.truncate(before='2018-01-01', after='2019-01-01')
s4_2018 = s4.truncate(before='2018-01-01', after='2019-01-01')
s5_2018 = s5.truncate(before='2018-01-01', after='2019-01-01')
s6_2018 = s6.truncate(before='2018-01-01', after='2019-01-01')
s7_2018 = s7.truncate(before='2018-01-01', after='2019-01-01')
s8_2018 = s8.truncate(before='2018-01-01', after='2019-01-01')
s9_2018 = s9.truncate(before='2018-01-01', after='2019-01-01')
s10_2018 = s10.truncate(before='2018-01-01', after='2019-01-01')


            
#%% 

s0_2018 = s0_2018["total_consumption_avg"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s1_2018 = s1_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s2_2018 = s2_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s3_2018 = s3_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s4_2018 = s4_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s5_2018 = s5_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s6_2018 = s6_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s7_2018 = s7_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s8_2018 = s8_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s9_2018 = s9_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()
s10_2018 = s10_2018["Total_MW"].sort_values(ascending=False).reset_index().drop("datetime", axis=1).dropna()

            
#%% 

s0_d = pd.DataFrame()
s1_d = pd.DataFrame()
s2_d = pd.DataFrame()
s3_d = pd.DataFrame()
s4_d = pd.DataFrame()
s5_d = pd.DataFrame()
s6_d = pd.DataFrame()
s7_d = pd.DataFrame()
s8_d = pd.DataFrame()
s9_d = pd.DataFrame()
s10_d = pd.DataFrame()

s0_d["d"] = 100*(s0_2018["total_consumption_avg"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"] 

s1_d["S1.NoTES.2018"] = 100*(s1_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s2_d["S2.NoTES.2018"] = 100*(s2_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]

s3_d["S3.TES=2h.2018"] = 100*(s3_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s4_d["S4.TES=4h.2018"] = 100*(s4_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s5_d["S5.TES=6h.2018"] = 100*(s5_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s6_d["S6.TES=12h.2018"] = 100*(s6_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]

s7_d["S7.TES=24h.2018"] = 100*(s7_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s8_d["S8.TES=36h.2018"] = 100*(s8_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s9_d["S9.TES=48h.2018"] = 100*(s9_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]
s10_d["S10.TES=60h.2018"] = 100*(s10_2018["Total_MW"]-s0_2018["total_consumption_avg"])/s0_2018["total_consumption_avg"]

  
#%% 

NoTES = pd.concat([s1_d["S1.NoTES.2018"],s2_d["S2.NoTES.2018"]],axis=1)
Hourly =pd.concat([s3_d["S3.TES=2h.2018"],s4_d["S4.TES=4h.2018"],s5_d["S5.TES=6h.2018"],s6_d["S6.TES=12h.2018"]],axis=1)
Daily = pd.concat([s7_d["S7.TES=24h.2018"],s8_d["S8.TES=36h.2018"],s9_d["S9.TES=48h.2018"],s10_d["S10.TES=60h.2018"]],axis=1)
  
  
#%% 

##FIGURE. Modification of power duration curve per scenario 

# Set the default color cycle
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r","indianred", "tomato", "navy","b","slateblue","mediumpurple"]) 

fig, axs = plt.subplots(4, sharex=False, sharey=False,figsize=(8, 8))
#fig.suptitle('Load aggregation strategies')
axs[0].plot(s0_2018)
axs[0].legend(["Baseline 2018","Baseline 2019"],loc='upper right',fontsize=9)
axs[0].set_ylim(10000, 60000)
axs[0].set_xlim(0, 8600)
axs[0].set_ylabel('Power (MW)')

axs[0].set_title("Load duration curve - Baseline",loc="left")


axs[1].plot(NoTES)
axs[1].legend(NoTES,loc='upper right',fontsize=8, ncol=2)
axs[1].set_ylim(0, 6)
axs[1].set_xlim(0, 8600)
axs[1].set_ylabel('Additional power (%)')
axs[1].set_title("No TES",loc="left")
axs[1].grid(axis='y')
axs[1].set_yticks(np.arange(0, 7, 1))

axs[2].plot(Hourly, label="actual")
axs[2].legend(Hourly,loc='upper right',fontsize=7, ncol=4)
axs[2].set_ylim(0, 6)
axs[2].set_xlim(0, 8600)
axs[2].set_ylabel('Additional power (%)')
axs[2].set_title("Hourly DR",loc="left")
axs[2].grid(axis='y')
axs[2].set_yticks(np.arange(0, 7, 1))

axs[3].plot(Daily, label="actual")
axs[3].legend(Daily,loc='upper right',fontsize=7, ncol=4)
axs[3].set_ylim(0, 6)
axs[3].set_xlim(0, 8600)
axs[3].set_xlabel('Hours')
axs[3].set_ylabel('Additional power (%)')
axs[3].set_title("Daily DR",loc="left")
axs[3].grid(axis='y')
axs[3].set_yticks(np.arange(0, 7, 1))

plt.margins(x=0.01)
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.92, wspace=None, hspace=0.45)
#plt.xticks(np.arange(0, 8600, 1000))
#plt.yticks(np.arange(0, 10, 2))

plt.show()


  
#%% 



#ADDITIONAL FIGURE. EXAMPLE OF HOURLY ENERGY CONSUMPTION MIX - INCLUDING ADDITIONAL WIND GENERATION

#DATA

# get current location of script
cwd = os.path.dirname(__file__) 
folder = "\data\INPUT_"+case_study #ES or GB
os.chdir(cwd+folder)

Listconsumption = ["datetime","power_consumption_nuclear_avg","power_consumption_geothermal_avg","power_consumption_biomass_avg","power_consumption_coal_avg",
                   "power_consumption_wind_avg","power_consumption_solar_avg","power_consumption_hydro_avg","power_consumption_gas_avg", "power_consumption_oil_avg",
                   "power_consumption_unknown_avg","power_consumption_battery_discharge_avg","power_consumption_hydro_discharge_avg"]

Listproduction= ["datetime","power_production_biomass_avg","power_production_coal_avg","power_production_gas_avg","power_production_hydro_avg","power_production_nuclear_avg","power_production_oil_avg","power_production_solar_avg","power_production_wind_avg","power_production_geothermal_avg","power_production_unknown_avg"]

df_MW = pd.read_csv(case_study+'_ElectricityMap_2018-2019.csv', index_col = "datetime", parse_dates=True,usecols=Listconsumption) 


df_co2 = pd.read_csv(case_study+'_ElectricityMap_2018-2019.csv', index_col = "datetime", parse_dates=True,usecols=("datetime","carbon_intensity_avg")) 
df_co2["carbon_intensity_avg"] = df_electricity["carbon_intensity_avg"]
df_co2["marginal_carbon_intensity_avg"] = df_electricity["marginal_carbon_intensity_avg"]

df_MW1 = pd.DataFrame()

df_MW1["Additional Wind"] = df_electricity["additional_wind"] 
df_MW1['Nuclear'] = df_MW['power_consumption_nuclear_avg']
df_MW1['Combustible fuels'] = df_MW['power_consumption_coal_avg']+df_MW['power_consumption_gas_avg']+df_MW['power_consumption_oil_avg']+df_MW["power_consumption_unknown_avg"]-df_MW1["Additional Wind"] 
df_MW1['Hydro'] = df_MW['power_consumption_hydro_avg']
df_MW1['Wind'] = df_MW["power_consumption_wind_avg"]
df_MW1['Solar'] = df_MW["power_consumption_solar_avg"]
df_MW1['Biomass'] = df_MW["power_consumption_biomass_avg"]
df_MW1['Geothermal'] = df_MW["power_consumption_geothermal_avg"]
df_MW1['Storage'] = df_MW["power_consumption_battery_discharge_avg"]+df_MW["power_consumption_hydro_discharge_avg"]




#%%

df_MW1_daily = df_MW1.truncate(before='2018-01-01', after='2018-01-15')
df_co2_t = df_co2.truncate(before='2018-01-01', after='2018-01-15')


#%%

# get current location of script
cwd = os.path.dirname(__file__) 

#Introduce folder of input data of selected case study
folder = "\Data\OUTPUT_"+case_study #ES or GB
os.chdir(cwd+folder)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rc('font',family='Arial',size=10)

styles7 = ["papayawhip", 'lightcyan',"cornflowerblue",'paleturquoise',"deepskyblue","steelblue","b","slategray",'lightcoral']


df_MW1_daily.plot.area(y=['Nuclear','Hydro','Additional Wind','Wind','Solar','Biomass','Geothermal','Storage','Combustible fuels'],
                       figsize=(12,3), ylim=[0,60000],style= styles7,linewidth=0,legend=True)
plt.title("Hourly energy consumption mix")
plt.xlabel("datetime(Days)")
plt.ylabel("MW")


ax = df_co2_t["carbon_intensity_avg"].plot(secondary_y=True, color='k',linestyle="--")
ax = df_co2_t["marginal_carbon_intensity_avg"].plot(secondary_y=True, color='gray',linestyle="--")
ax.set_ylabel("gCO2eq/kWh")
ax.set_ylim(0,600)

plt.show()

