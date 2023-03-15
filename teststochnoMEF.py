# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:32:08 2022

@author: HP
"""

from borg import *
import numpy as np
import platform  # helps identify directory locations on different types of OS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
import sys
from sys import *
from scipy.optimize import root
import scipy.stats as ss
from numpy.core.multiarray import ndarray
import sys
import NSD_objectives_optim_dailystoch_noMEF
from NSD_objectives_optim_dailystoch_noMEF import deterministic
import time
start_time = time.time()

nvars = 2
nobjs = 4
k = nvars - nobjs + 1

#QD = pd.read_csv('Godavari_10k.csv', header=None, dtype=float)
#QR = pd.read_csv('Krishna_10k.csv', header=None, dtype=float)

start_time = time.time()

def Simulation_Caller(*vars):
    '''
    Purpose: Borg calls this function to run the simulation model and return multi-objective performance.

    Note: You could also just put your simulation/function evaluation code here.

    Args:
        vars: A list of decision variable values from Borg

    Returns:
        performance: policy's simulated objective values. A list of objective values, one value each of the objectives.
    '''

    borg_vars = vars  # Decision variable values from Borg

    # Reformat decision variable values as necessary (.e.g., cast borg output parameters as array for use in simulation)
    #op_policy_params = np.asarray(borg_vars)
    # Call/run simulation model, return multi-objective performance:

    #performance = pysedsim.PySedSim(decision_vars=op_policy_params) ##altered by veena
    
    performance = deterministic(borg_vars)
    return performance

print("---run_time %s seconds ---" % (time.time() - start_time))
#result= deterministic(np.zeros((1,12)))
#print(result)
nSeeds = 1 
for j in range(nSeeds):

 borg = Borg(nvars, nobjs, 0, Simulation_Caller)
 borg.setBounds([-1, 1],[0,1])
 borg.setEpsilons(50,50,0.01,0.01)

     # Define the filepath for each runtime file

 runtime_filename = os.getcwd() + '/runtime2_stochnoMEF_40/' + 'runtime_file_seed_' + str(j + 1) + '.runtime'

 result = borg.solve({"maxEvaluations": 50000, "runtimeformat": 'borg', "frequency": 500,
                          "runtimefile": runtime_filename})

 f = open(os.getcwd() + '/sets/' + str(j + 1) + '.txt', 'w')

 f.write('#Borg Optimization Results\n')
 f.write('#First ' + str(nvars) + ' are the decision variables, ' \
                                      'last ' + str(nobjs) + ' are the objective values\n')

 for solution in result:
     line = ''
     for i in range(len(solution.getVariables())):
         line = line + (str(solution.getVariables()[i])) + ' '

     for i in range(len(solution.getObjectives())):
         line = line + (str(solution.getObjectives()[i])) + ' '

     f.write(line[0:-1] + '\n')

 f.write("#")

 f.close()
 #j = int(sys.argv[0])
 
#for j in range(nSeeds):
# j = int(sys.argv[1])
# borg = Borg(nvars, nobjs, nconstrs, deterministic)
# borg.setBounds(*[[0, 2800]] * (nvars))
# borg.setEpsilons(0.01,0.01,0.01,0.01,0.01)
     # Define the filepath for each runtime file

# runtime_filename = os.getcwd() + '/runtime/' + 'runtime_file_seed_' + str(j + 1) + '.runtime'

# result = borg.solve({"maxEvaluations": 2000, "runtimeformat": 'borg', "frequency": runtime_freq,
 #                         "runtimefile": runtime_filename})

 #f = open(os.getcwd() + '/sets/' + \
#              str(j + 1) + '.set', 'w')

# f.write('#Borg Optimization Results\n')
# f.write('#First ' + str(nvars) + ' are the decision variables, ' \
#                                      'last ' + str(nobjs) + ' are the objective values\n')

# for solution in result:
#     line = ''
#     for i in range(len(solution.getVariables())):
#         line = line + (str(solution.getVariables()[i])) + ' '
#
#     for i in range(len(solution.getObjectives())):
#         line = line + (str(solution.getObjectives()[i])) + ' '

#     f.write(line[0:-1] + '\n')

# f.write("#")

# f.close()

print("---run_time %s seconds ---" % (time.time() - start_time))