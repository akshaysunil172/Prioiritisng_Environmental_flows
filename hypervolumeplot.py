# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 22:20:46 2023

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
  

  
# using the style for the plot
plt.style.use('ggplot')
  

hypervolume  = np.loadtxt(open("hypervolume.csv"), delimiter=",")

hypervol = hypervolume[:,1:6];
hypervol = hypervol/np.max(hypervol);
NFE = hypervolume[:,0];

plt.plot(NFE,hypervol)

plt.rcParams["figure.figsize"] = [5, 3]
#plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
#plt.rcParams["axes.linewidth"] = 2.50

plt.ylabel('Relative Hypervolume', fontsize = 10)
plt.xlabel('Number of Functional Evaluations', fontsize = 10)

plt.legend(['Seed 1','Seed 2','Seed 3','Seed 4','Seed 5'], fontsize = 8)

plt.tight_layout()
plt.savefig('hypervolume_plot.png', dpi = 1200)