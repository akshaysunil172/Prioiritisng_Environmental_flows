# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:54:28 2022

@author: HP
"""

"""
Created on Sun Apr 24 01:10:57 2022

@author: Akshay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sys import *
from scipy.optimize import root
import scipy.stats as ss
#from borg import *
from numpy.core.multiarray import ndarray
import sys
from joblib import Parallel, delayed
import multiprocessing
import time

#Q2h = pd.read_csv("Q2h_NS_1968.csv");
#inflowdata = np.array(Q2h);
#inflow = inflowdata[:, 3];
#inflow_month = inflowdata[:, 1];
#inflow_month = inflow_month[0:365];
#inflow_month = np.array(inflow_months)

inflow_month = pd.read_csv("months.csv");
inflow_month = np.array(inflow_month);
inflow_month = inflow_month[:,0];

#QD = pd.read_csv("NS-10000x10-daily.csv");
QD = pd.read_csv("NS_30yearnew-10000x10-daily.csv");
QD = np.array(QD)


PE_daily = pd.read_csv("PE_daily.csv");
PE_daily = np.array(PE_daily)

#param = [0.5,0.7];

#Flood thresholds for calculating flood reliability

#max_releases = pd.read_csv("flood_threshold.csv");
#max_releases = np.array(max_releases);
#max_releases = max_releases[0:3650, 3];

#plt.plot(max_releases)

max_releases = pd.read_csv("flood_threshold_calendaryear.csv");
max_releases = np.array(max_releases);
max_releases = max_releases[:, 1];

#plt.plot(max_releases)



#MEF thresholds for calculatingf MEF reliability
MEF_thresholds = pd.read_csv("Mean_inflows_calendaryear.csv");
MEF_thresholds = np.array(MEF_thresholds);
MEF_thresholds = MEF_thresholds[:, 1]*0.4;

# Potential Evapotranspiration


n_year = 10;
n_days = 1;
livestorage = 5733;


#os.chdir('D:\optimizerbfs_python);

demand_data = pd.read_csv("Demand_calendaryear.csv");
demand_data= np.array(demand_data)

# Q2h is in cubic feet/sec, convert to Mm3

#demand represents the total demand

demand = demand_data[:,1];
demand_left = demand_data[:,1]/2
demand_right = demand_data[:,1]/2
HPcap_main = 810; # MW
HPcap_left = 60;
HPcap_right = 90;
dis_cap_right      = 423.81*3600*24/np.power(10,6); #converted m3/s to Mm3
dis_cap_left       = 423.81*3600*24/np.power(10,6); #converted m3/s to Mm3
dis_cap_main       = 1168*3600*24/np.power(10,6);   #converted m3/s to Mm3
canal_cap_left     = 311.5*3600*24/np.power(10,6);  #converted m3/s to Mm3
canal_cap_right    = 311.5*3600*24/np.power(10,6);  #converted m3/s to Mm3
INS_canal_cap      = 1090*3600*24/np.power(10,6);   #converted m3/s to Mm3
A0                 = 186; #Mm2
a                  = 295/5733; #Mm2
et                 = PE_daily[:,3]/1000; # m/day
#n_data = len(inflow);
n_data = 3650;


# maximum allowable discharge to hydropower plant 
# main 1168 m3/s, right and left 423.81 m3/s
# Leelakrishna et al 2019: left and right canal capacity 311.5 m3/s
# INS canal capacity 1090m3/s

tailwater_left     = 155; #m
tailwater_right    = 145; #m
tailwater_main     = 73; #m
efficiency = 0.8;


## Defining empty vectors to save computational time inside the loop 
Head            = np.zeros((n_data));
storage         =  np.zeros((n_data));
storage_pre_releases = np.zeros((n_data));
supply          =  np.zeros((n_data));
MEF_releases    =  np.zeros((n_data));
deficit         =  np.zeros((n_data));
HPL             =  np.zeros((n_data));
HPR             =  np.zeros((n_data));
HPM             =  np.zeros((n_data));
spill           =  np.zeros((n_data));
norm_stor       =  np.zeros((n_data));
release_main    =  np.zeros((n_data));
release_left    =  np.zeros((n_data));
release_right   = np.zeros((n_data));
act_release_main = np.zeros((n_data));
act_release_flood = np.zeros((n_data));
HPM_effi        =  np.zeros((n_data));
HPL_effi        =  np.zeros((n_data));
HPR_effi        =  np.zeros((n_data));
act_release_main_withSpill    =   np.zeros((n_data));
total_downstream_releases =  np.zeros((n_data));
spill_act       =   np.zeros((n_data));
rbf =  np.zeros((n_data));
EL         =   np.zeros((n_data));
param = np.zeros(2)




def deterministic(vars):
    nobjs = 4
    param = vars;
    #nvars = 12
    nconstrs = 0
    #n = 15
    
    Q1 = np.ones((5, 3650))*10000

    #desc_var = vars
    #desc_var = np.tile(np.asarray(vars), n)
    #print(desc_var)
    for i in range(5):
        #np.random.seed(i+np.random.randint(0,10000))
        Q1[i] = QD[np.random.randint(0,9999)]

    objectives = np.zeros((5, 4))

    num_cores = 14
    results = Parallel(n_jobs=num_cores)(delayed(eval_obj)(Q1[s], param) for s in range(5))
    objectives = [x[0] for x in results]
    mean_objectives = np.zeros(4)
    #mean_constraints = np.zeros(2)
    mean_objectives = np.mean(objectives,axis= 0)
    #mean_constraints = np.mean(constraints,axis= 0)
    
    objs = [0.0] * nobjs
    objs[0] = mean_objectives[0]
    objs[1] = mean_objectives[1]
    objs[2] = mean_objectives[2]
    objs[3] = mean_objectives[3]



    #objectives = np.zeros((10, 4))


   # num_cores = 14
   # results = Parallel(n_jobs=num_cores)(delayed(eval_obj)(param)for s in range(1))[0]   


    
    #objs = [0.0] * nobjs
   # objs[0] = results[0][0]
   # objs[1] = results[0][1]
   # objs[2] = results[0][2]
   # objs[3] = results[0][3]
    

    result_f =  (objs)
    return(result_f)

# System model

def eval_obj(Q1, param):
    
  #start_time = time.time()
    
  for i in range(0, n_data):
    
    # Add supply to reservoir volume

    if i != 0:
        storage[i] = storage[i - 1] + Q1[i]
    else:
        storage[i] = storage[i] + Q1[i]

    #  Remove MEF requirements
    #MEF_releases[i] = min(MEF_thresholds[i],storage[i]);
    
    # Update storage by subracting the MEF releases made from reservoir
    #storage[i] = storage[i] - MEF_releases[i];

    # Remove Demands

    supply[i] = min(demand[i], storage[i]) ; #canalcap1+canalcap2

    # Update storage by subracting the supply made from reservoir 

    storage[i] = storage[i] - supply[i];
 
     
    #Calculate deficits
    
    deficit[i] = demand[i] - supply[i];
    
    # calculate releases made to left and right canals
    
    release_left[i] = supply[i]*demand_left[i]/demand[i];
    release_right[i] = supply[i]*demand_right[i]/demand[i];

    # Calculate hydropwer generated using Hydropower function
    
    HPL[i],HPL_effi[i],Head[i] = Hydropower(storage[i],supply[i]*demand_right[i]/demand[i],HPcap_right,dis_cap_right,tailwater_right,n_days);
    
    HPR[i],HPR_effi[i],Head[i] = Hydropower(storage[i],supply[i]*demand_left[i]/demand[i],HPcap_left,dis_cap_left,tailwater_left,n_days);
       
    
    # Calculate hydropower generated using Hydropower function

   # print("---run_time %s seconds ---" % (time.time() - start_time))
    # Calaculate releases made to downstream 

    norm_stor[i] = storage[i] /(livestorage + max(Q1)); 
    storage_pre_releases[i] = storage[i];

    release_main[i] = dis_cap_main* np.exp(-((np.square(norm_stor[i] - param[0]) / np.square(param[1]))))
    
    act_release_main[i] =  min(release_main[i], storage[i]) ;
    
    #  Adding MEF releases to compute flood reliability
    act_release_flood[i]= act_release_main[i]+MEF_releases[i];
    
    # Update storage by removing releases made to downstream  
    
    storage[i] = storage[i] - act_release_main[i];

    #Remove spills
    spill[i] = max(0,storage[i]- livestorage);
    storage[i] = storage[i] - spill[i];
    act_release_main_withSpill[i] = min(act_release_main[i] + spill[i],dis_cap_main);
    total_downstream_releases[i] = act_release_main[i] + spill[i];
    #spill_act = spill[i] - act_release_main_withSpill[i];   
     
    # Main turbine hydropower generated by downstream releases
    
    HPM[i],HPM_effi[i],Head[i] = Hydropower(storage[i],act_release_main_withSpill[i],HPcap_main,dis_cap_main,tailwater_main,n_days);
    
    # Removing Evapotranspiration losses
    
    
    if i != 0:
        EL[i] = A0*et[i] + 0.5*a*et[i] *(storage[i]+storage[i-1]);

    else:
        EL[i] = A0*et[i] + 0.5*a*et[i] *(storage[i]+0);
        
        
    ## Calculation of objectives
    # Objectives
    # 1. Maximize Hydropower
    objs = np.zeros(4);
    objs[0] = -(np.sum(HPL)+np.sum(HPR)+np.sum(HPM))/n_year; #In MU, ie. MKWh
    
    # 2. Minimize Deficits
    objs[1] = np.sum(deficit)/n_year;
    
    # 3. Maximize flood Reliability
    # indtouse is the integer for flood months 
    
    indtouse = np.where((inflow_month >= 7) & (inflow_month <= 9));
    
    #flood values for flood months 
    floodtouse1 = total_downstream_releases[indtouse];
    maxin_donor_touse = max_releases[indtouse];
    total_flood_exceedences = np.where(floodtouse1>maxin_donor_touse)
    flood_fail_ratio = (np.size(total_flood_exceedences,1))/len(floodtouse1);
    objs[2] = -(1-flood_fail_ratio);
    
    # 4 . Maximize MEF Reliability

    meffailure_ratio  = np.size(np.where(total_downstream_releases < MEF_thresholds))/len(total_downstream_releases);
    objs[3] = -(1-meffailure_ratio);
    
#    objs = np.array(objs).tolist();
    
  return[objs];    

#plt.plot(total_downstream_releases)
#plt.plot(MEF_thresholds)
    
def Hydropower(St,R,HPCapacity,dis_cap,tailwater,days):
    
    efficiency = 0.8;
    #Ht = np.zeros((n_data));
    #Ht_net = np.zeros((n_data));
    #P = np.zeros((n_data));
    #Pb_effi = np.zeros((n_data));
    
    #for i in range(0, n_data):
    Ht= 0.004*St+156.3;
    Ht_net = abs(Ht - tailwater) #Net Head
    
    
    HPCapacity_in_timeperiod = HPCapacity/1000*days*24;
    
    
    #R_cap = np.zeros(n_data)
    #P = np.zeros(n_data)
    
    #for i in range(0, n_data):
    R_cap = min((R, dis_cap));
    
   #Power generated
    
    #for i in range(0, n_data):
    P = min(efficiency*2725*Ht_net*R_cap/1000/1000,HPCapacity_in_timeperiod);
    Pb_effi = min(efficiency*2725*Ht_net*R/1000/1000, HPCapacity_in_timeperiod)/HPCapacity_in_timeperiod*100;
   
    return(P, Pb_effi,Ht)

