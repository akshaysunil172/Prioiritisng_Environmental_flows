# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:48:39 2022

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:00:57 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import pandas
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

#fig = plt.figure()

#####################################################################################################################
#to have transprency based on transfer volume values
# data1 = (pd.read_csv('act_var_int_month.csv',header=None,dtype=float))
# dataf1 = np.sum((data1[data1.columns[0:12]]),axis=1)
# data1_ann = dataf1.groupby(dataf1.index // 15).mean()

# data2 = (pd.read_csv('act_var_dps1_month.csv',header=None,dtype=float))
# dataf2 = np.sum((data2[data2.columns[0:12]]),axis=1)
# data2_ann = dataf2.groupby(dataf2.index // 15).mean()

# data3 = (pd.read_csv('act_var_dps2_month.csv',header=None,dtype=float))
# dataf3 = np.sum((data3[data3.columns[0:12]]),axis=1)
# data3_ann = dataf3.groupby(dataf3.index // 15).mean()


# alpha_data = data1_ann.append(data2_ann)
# alpha_data = alpha_data.append(data3_ann)

# data_alpha = np.asarray(alpha_data)
# norm_alpha = data_alpha.copy()
# min_alpha = np.min(data_alpha)
# max_alpha = np.max(data_alpha)
# for i in range(167):
#     norm_alpha[i] = 1 - (data_alpha[i] - min_alpha) / (max_alpha - min_alpha)
# print(norm_alpha)


###############################################################################################################
####################################To plot Parallel coordinate################################################
###############################################################################################################
fig = plt.figure()
ax2 = fig.add_subplot(411)

# to remove outer box
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

text_pos_x = 0.06
text_pos_y = 0.952
plt.text(text_pos_x, text_pos_y, "(a)", fontsize=16,
transform=plt.gcf().transFigure, color='black')

text_pos_xn = 0.06
text_pos_yn = 0.68
plt.text(text_pos_xn, text_pos_yn, "(b)", fontsize=16,
transform=plt.gcf().transFigure, color='black')

plt.text(0.36, 0.68, "(c)", fontsize=16,
transform=plt.gcf().transFigure, color='black')

#plt.text(0.36, 0.68, "(c)", fontsize=16,
#transform=plt.gcf().transFigure, color='black')



#ax3 = fig.add_subplot(212)

###########################################
#to plot parallel coordinates plot
###########################################
ax22 = ax2.twiny()

ax22.spines['top'].set_visible(False)
ax22.spines['right'].set_visible(False)
ax22.spines['left'].set_visible(False)
ax22.spines['bottom'].set_visible(False)

#xticks = [0.075, 0.725,1.375, 2.025, 2.675]
xticks = [0.5,1.5,2.5,3.5]
genmax = [4000,2100, 100,100]
genmin = [0, 0, 0, 0]

data_MEF  = np.loadtxt(open("objs_MEF.csv"), delimiter=",")

data_MEF_paretosorted = np.loadtxt(open("objs_MEF_pareto_set.csv"), delimiter=",")

#                        # names=['RelD','RelR','ResD','ResR','VulD','VulR','FRD','FRR','EFD','EFR'])
data_noMEF  = np.loadtxt(open("objs_noMEF.csv"), delimiter=",")

data_noMEF_paretosorted = np.loadtxt(open("objs_noMEF_pareto_set.csv"), delimiter=",")

data_hist_det = np.loadtxt(open("objs_historical.csv"), delimiter=",")

# #data2  = pandas.read_table("reeval_DPS2.txt",
#    #                      sep=' ', header=None)
# all_data = pandas.read_table("reeval_combined.txt",sep = ' ',header = None)
#utils = ['RelD','RelR','ResD','ResR','VulD','VulR','FRD','FRR','EFD','EFR']
#all_data = np.loadtxt(open("all_data.csv"), delimiter=",")

all_data = np.append(data_MEF,data_noMEF, axis=0)
all_data_sorted = np.append(data_MEF_paretosorted,data_noMEF_paretosorted, axis=0)
all_data = np.append(all_data,all_data_sorted, axis=0)
all_data = np.append(all_data,data_hist_det, axis=0)

#import pylab as p
#p.arrow( x, y, dx, dy, **kwargs )
#p.arrow( 0.3,0.8, 0.0,2, fc="k", ec="k",head_width=0.05, head_length=0.1 )

m = len(data_MEF)
n = len(data_noMEF)
o = len(data_MEF_paretosorted)
p = len(data_noMEF_paretosorted)


#for i in range(167):
 #   norm_alpha1[i] = 1 - (data_alpha1[i] - min_alpha1) / (max_alpha1 - min_alpha1)
#print(norm_alpha1)

#for i in range(167):
 #   norm_alpha2[i] = 1 - (data_alpha2[i] - min_alpha2) / (max_alpha2 - min_alpha2)
#print(norm_alpha2)

objectives = all_data

#normalisation of objectives
mm = objectives.min(axis=0)
mx = objectives.max(axis=0)
mm[0] = min(objectives[:,0])*1
mm[1] = min(objectives[:,1])*1
mm[2] = min(objectives[:,2])*100
mm[3] = min(objectives[:,3])*100


mx[0] = max(objectives[:,0])*1
mx[1] = max(objectives[:,1])*1
mx[2] = max(objectives[:,2])*100
mx[3] = max(objectives[:,3])*100


mean_objectives = np.zeros([4,len(objectives)])
mean_objectives[0] = (objectives[:,0])
mean_objectives[1] = (objectives[:,1])
mean_objectives[2] = (objectives[:,2])*100
mean_objectives[3] = (objectives[:,3])*100

norm_objectives = mean_objectives.copy()

for i in range(4):
    #mm = objectives[:,i].min()
    #mx = objectives[:,i].max()
    if all(mean_objectives[i,:]==mean_objectives[i,1]):
        mm[i] = genmin[i] #mins
        mx[i] = genmax[i] #maxs

    if mm[i]!=mx[i]:
        if i==0 or i==2 or i == 3:
            norm_objectives[i,:] = (norm_objectives[i,:] - mm[i]) / (mx[i] - mm[i])
        else:
            norm_objectives[i,:] = (norm_objectives[i,:] - mx[i]) / (mm[i] - mx[i])
    else:
        norm_objectives[i,:] = 1

xs = xticks

for i in range(0,1):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#5DC83F', alpha=1, linewidth=1.6) 
    
for i in range(n,n+1):
    ys = (norm_objectives[:,i])
    xs = xticks
    line2 = ax2.plot(xs, ys,color='#DC143C', alpha=1, linewidth=1.6)
    
for i in range(m+n+o+p,m+n+o+p+1):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='black', alpha=0.8, linewidth=1.6)

for i in range(m+n+o+p+1,m+n+o+p+2):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#5DC83F', alpha=1, linewidth=1.6,linestyle='dashed')
  
for i in range(m+n+o+p+2,m+n+o+p+3):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#DC143C', alpha=1, linewidth=1.6,linestyle='--')

for i in range(m+n+o+p+2,m+n+o+p+3):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#DC143C', alpha=1, linewidth=1.6,linestyle='--')

for i in range(n,m+n):
    ys = (norm_objectives[:,i])
    xs = xticks
    line2 = ax2.plot(xs, ys,color='#DC143C', alpha=0.2, linewidth=1.6)

#Actual plotting code starts

for i in range(0,m):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#5DC83F', alpha=0.2, linewidth=1.5)     
    
for i in range(m+n,m+n+o):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#5DC83F', alpha=0.9, linewidth=1.5)
    
for i in range(m+n+o,m+n+o+p):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#DC143C', alpha=0.8, linewidth=1.5)
    
for i in range(m+n+o+p,m+n+o+p+1):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='black', alpha=0.8, linewidth=1.5)
    
for i in range(m+n+o+p+1,m+n+o+p+2):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#5DC83F', alpha=1, linewidth=1.5,linestyle='dotted')
    
for i in range(m+n+o+p+2,m+n+o+p+3):
    ys = (norm_objectives[:,i])
    xs = xticks
    line1= ax2.plot(xs, ys,color='#DC143C', alpha=1, linewidth=1.5,linestyle='dotted')

# for i in range(len(data1[:].values)+len(data[:].values)+2,len(objectives)):
#     ys = (norm_mean_objectives[:,i])
#     xs = xticks
#     ax2.plot(xs, ys,c =(0.12,0.47,0.71), alpha=norm_alpha[i-, linewidth=2)
plt.arrow(0.35, 0.15,0, 0.8, clip_on=False,width=0.005,color='black', length_includes_head=True,fc="k", ec="k",head_width=0.04, head_length=0.05)



pop_a = mpatches.Patch(color='#5DC83F', label='PF_MEF', linewidth= 0.1)
pop_b = mpatches.Patch(color='#DC143C', label='PF_nMEF', linewidth= 0.1)
pop_c = mpatches.Patch(color='black', label='Histocial', linewidth= 0.1)
pop_d = mpatches.Patch(color='black', label='Deterministic', linewidth= 0.05,linestyle='--')

#pop_d = mpatches.Patch(color='#1E1E1E', label='P2 (not prioritising MEF)')
#plt.legend(handles=[pop_a,pop_b])

green_star = mlines.Line2D([], [],c ='None',markerfacecolor ='green', marker='*', linestyle='None',
                          markersize=12, label='Policy [PF_MEF]')

red_star = mlines.Line2D([], [], c ='None',markerfacecolor ='red',marker='*', linestyle='None',
                          markersize=12, label='Policy [PF_nMEF]')




#ax2.legend(['MEF ','No MEF'],handles=[pop_a,pop_b],loc='lower center',ncol = 2,fontsize=14,frameon=False,bbox_to_anchor=(0.25, -0.2))
ax2.legend(['PF_MEF ','PF_nMEF', 'Historical','Deterministic [PF_MEF]','Deterministic [PF_nMEF]'],loc='best',ncol = 1,fontsize=13,frameon=False,bbox_to_anchor=(1.079, 0.775))
ax22.legend([' ',''],handles=[green_star,red_star],loc='best',ncol = 1,fontsize=13,frameon=False,bbox_to_anchor=(1.444, 1))
#ax2.legend(['Fortnightly (FN)','Annual (AN)'],loc='lower center',ncol = 2,fontsize=12,bbox_to_anchor=(0.5, 0.15))

# plotting markers

#plt.legend(handles=[blue_star])
#ax2.plot(xs, norm_mean_objectives[:,14],c ='None',markerfacecolor = 'grey',markeredgecolor='None',markersize = 12,marker='p')
#ax2.plot(xs, norm_mean_objectives[:,5],c ='None',markerfacecolor ='black',markersize = 12,marker='*')


##ax2.plot(xs, norm_mean_objectives[:,25],c ='None',markerfacecolor ='black',markersize = 12,marker='s')
##ax2.plot(xs, norm_mean_objectives[:,7],c ='None',markerfacecolor ='black',markersize = 12,marker='*')
#ax2.plot(xs, norm_mean_objectives[:,24],c ='None',markerfacecolor ='black',markersize = 11,marker='p')
ax2.plot(xs, norm_objectives[:,28],c ='None',markerfacecolor ='red',markersize = 12,marker='*')
ax2.plot(xs, norm_objectives[:,8],c ='None',markerfacecolor ='green',markersize = 12,marker='*')
##ax2.plot(xs, norm_mean_objectives[:,14],c ='None',markerfacecolor ='black',markersize = 12,marker='p')

# ax2.plot(xs, norm_mean_objectives[:,0],c ='None',markerfacecolor = 'grey',markeredgecolor='None',markersize = 12,marker='p')
# ax2.plot(xs, norm_mean_objectives[:,1],c ='None',markerfacecolor='black',markersize = 12,marker='*')


ax2.set_yticks([ ])
ax2.set_ylim([-0.02,1.02])
ax2.set_xticks(xticks)
xticklabels = ["{0:.0f}".format(mm[0]),"{0:.0f}".format(mx[1]),"{0:.0f}".format(mm[2]),"{0:.0f}".format(mm[3])]
ax2.set_xticklabels(xticklabels, fontsize=16)
ax2.set_xlim([0.45,4-0.45])



ax22.set_xticks(xticks)
xticklabels_top = ["{0:.0f}".format(mx[0]), "{0:.0f}".format(mm[1]),"{0:.0f}".format(mx[2]),"{0:.0f}".format(mx[3])]
ax22.set_xticklabels(xticklabels_top, fontsize=16)
ax22.set_xlim([0.45,4-0.45])

# ax32 = ax3.twiny()
#ax22.set_xlabel('Hydropower[GWh]     Average Deficit[MCM]                     Flood Reliability[%]                  MEF Reliability[%]',position=(0.005, 1e1),horizontalalignment='left', fontsize =16)
ax22.set_xlabel('Hydropower[GWh]    Average Deficit[MCM]    Flood Reliability[%]     MEF Reliability[%]',position=(-0.12, 1e1),horizontalalignment='left', fontsize =14)


ax2.grid(True, linewidth = "1.2", color = "black")

#ax3.grid(True)

#adding texts a and b

#plt.figtext(0.03, 0.95, 'a)',verticalalignment='bottom', horizontalalignment='left', fontsize=12)
#plt.figtext(0.03, 0.5, 'b)',verticalalignment='bottom', horizontalalignment='left', fontsize=12)

####################################################
#creating legend for opaqueness in the figure
# cmap = mpl.colors.ListedColormap([(0,0,0,0), (0,0,0,0.5),(0,0,0,1)])
# ax1 = fig.add_axes([0.1, 0.08, 0.3, 0.02])
# bounds = [0.0, 0.5, 1.0]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# cb3 = mpl.colorbar.ColorbarBase(ax1, cmap='Greys', extendfrac='auto',
#                                ticks=bounds,
#                               spacing='uniform',
#                              orientation='horizontal')
# cb3.set_ticklabels([str(int(max_alpha)) + ' Mm$^3$', str(int((max_alpha+min_alpha)/2)) + ' Mm$^3$', str(int(min_alpha))+ ' M$m^3$'])
# cb3.set_label('Mean transfer volumes [Transparency]')
#print(len(data[:].values))
######################################################
#ax1.annotate("", xy=(0.8, 0.44), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))
#######################################################
fig.set_size_inches([12, 16])


#x = np.linspace(0, 2 * np.pi, 400)
#y = np.sin(x ** 2)
ax3 = fig.add_subplot(423)



#for i in range(0,m):
    #ys = (norm_mean_objectives[:,i])
    #xs = xticks
scatter1 = ax3.scatter(data_MEF[:,1],data_MEF[:,0],color='#5DC83F', alpha=0.2, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF[:,1],data_noMEF[:,0],color='#DC143C', alpha=0.2, linewidth=1.5) 
scatter3 = ax3.scatter(data_hist_det[0,1],data_hist_det[0,0],color='black', alpha= 1, linewidth=1.5)
#scatter3 = ax3.scatter(all_data[28,1],all_data[28,0],color='black', alpha= 1, linewidth=1.5)
#scatter4 = ax3.scatter(all_data[28,1],all_data[28,0],color='blue', alpha= 1, linewidth=1.5) 

plt.xlabel("Average Deficit [MCM]", fontsize=14)
plt.ylabel("Hydropower [GWh]" , fontsize=14)
ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
    #ax3.legend("PF_MEF","PF_nMEF")
ax3.invert_xaxis()

arrow_pos_1  = 0.09
arrow_pos_2  = 0.455
arrow_pos_3  = 0.07
ax3.text(0.2, 0.445, r'$\;\rightarrow$', fontsize=24,
transform=plt.gcf().transFigure, color='black')
plt.text(0.015, 0.56, r'$\;\uparrow$', fontsize=24,
transform=plt.gcf().transFigure, color='black')
plt.text(0.36, 0.44, "(e)", fontsize=16,
transform=plt.gcf().transFigure, color='black')

plt.text(0.06, 0.44, "(d)", fontsize=16,
transform=plt.gcf().transFigure, color='black')

#ax3.annotate('', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1),
#arrowprops=dict(arrowstyle="<->", color='b'))
#ax3.arrow(1500, 3000, 0, 1000, head_width=40, head_length=80, fc='k')
#ax3.annotate('', xy=(0.1, -0.1), xycoords='axes fraction', xytext=(1, -0.1), 
       #     arrowprops=dict(arrowstyle="<-", color='black'))



    #ax.invert_yaxis()
   # fig.set_size_inches([12, 10])

#for i in range(m+n,m+n+o):
    #ys = (norm_mean_objectives[:,i])
    #xs = xticks
scatter1 = ax3.scatter(data_MEF_paretosorted[:,1],data_MEF_paretosorted[:,0],color='#5DC83F', alpha=1, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF_paretosorted[:,1],data_noMEF_paretosorted[:,0],color='#DC143C', alpha=1, linewidth=1.5)
scatter4 = ax3.scatter(all_data[28,1],all_data[28,0],color='black', marker='*', linestyle='None', alpha= 1,  s=350)
scatter5 = ax3.scatter(all_data[28,1],all_data[28,0],color='green', marker='*', linestyle='None', alpha= 1,  s=120)
scatter4 = ax3.scatter(all_data[8,1],all_data[8,0],color='black', marker='*', linestyle='None', alpha= 1,  s=350)
scatter5 = ax3.scatter(all_data[8,1],all_data[8,0],color='red', marker='*', linestyle='None', alpha= 1, s=120)   

plt.xlabel("Average Deficit [MCM]", fontsize=14)
plt.ylabel("Hydropower [GWh]" , fontsize=14)
ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)

#ax3.arrow(0.35, 0.4,0, 0.5, clip_on=False,width=0.005,color='black', length_includes_head=True,fc="k", ec="k",head_width=0.04, head_length=0.05)

#plt.arrow(0.1, 0.15,0, 1, clip_on=False,width=0.008,color='black', length_includes_head=True)

#ax3.legend("PF_MEF","PF_nMEF")
#ax3.invert_xaxis()

#for i in range(m+n,m+n+o):
#    ys = (norm_mean_objectives[:,i])
  #  xs = xticks
  #  line1= ax2.plot(xs, ys,color='#5DC83F', alpha=0.9, linewidth=1.5)

    
ax3 = fig.add_subplot(424)


scatter1 = ax3.scatter(data_MEF[:,1],data_MEF[:,3]*100,color='#5DC83F', alpha=0.2, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF[:,1],data_noMEF[:,3]*100,color='#DC143C', alpha=0.2, linewidth=1.5)
scatter3 = ax3.scatter(data_hist_det[0,1],data_hist_det[0,3]*100,color='black', alpha= 1, linewidth=1.5)

ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
plt.xlabel("Average Deficit [MCM]", fontsize=14)
plt.ylabel("MEF Reliability [%]" , fontsize=14)

ax3.invert_xaxis()

scatter1 = ax3.scatter(data_MEF_paretosorted[:,1],data_MEF_paretosorted[:,3]*100,color='#5DC83F', alpha=1, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF_paretosorted[:,1],data_noMEF_paretosorted[:,3]*100,color='#DC143C', alpha=1, linewidth=1.5) 
scatter4 = ax3.scatter(all_data[28,1],all_data[28,3]*100,color='black', marker='*', linestyle='None', alpha= 1,  s=350)
scatter4 = ax3.scatter(all_data[8,1],all_data[8,3]*100,color='black', marker='*', linestyle='None', alpha= 1,  s=350)
scatter6 = ax3.scatter(all_data[28,1],all_data[28,3]*100,color='red', marker='*', linestyle='None', alpha= 1, s=120)
scatter7 = ax3.scatter(all_data[8,1],all_data[8,3]*100,color='green', marker='*', linestyle='None', alpha= 1, s=120) 

plt.xlabel("Average Deficit [MCM]", fontsize=14)
plt.ylabel("MEF Reliability [%]"  , fontsize=14)
ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
#scatter3 = ax3.scatter(data_hist_det[1,1],data_hist_det[1,3],color='black', alpha= 1, linewidth=1.5) 

ax3 = fig.add_subplot(425)


scatter1 = ax3.scatter(data_MEF[:,0],data_MEF[:,3]*100,color='#5DC83F', alpha=0.2, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF[:,0],data_noMEF[:,3]*100,color='#DC143C', alpha=0.2, linewidth=1.5)
scatter3 = ax3.scatter(data_hist_det[0,0],data_hist_det[0,3]*100,color='black', alpha= 1, linewidth=1.5)


ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
plt.xlabel("Hydropower [GWh]", fontsize=14)
#plt.set_xlabel(r'$\rho/\rho_{ref}\;\rightarrow$', color='red')
#plt.set_ylabel(r'$\Delta \Theta / \omega \longrightarrow$')
plt.ylabel("MEF Reliability [%]" , fontsize=14)   

scatter1 = ax3.scatter(data_MEF_paretosorted[:,0],data_MEF_paretosorted[:,3]*100,color='#5DC83F', alpha=1, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF_paretosorted[:,0],data_noMEF_paretosorted[:,3]*100,color='#DC143C', alpha=1, linewidth=1.5)

scatter4 = ax3.scatter(all_data[28,0],all_data[28,3]*100,color='black', marker='*', linestyle='None', alpha= 1,  s=350)
scatter5 = ax3.scatter(all_data[8,0],all_data[8,3]*100,color='black', marker='*', linestyle='None', alpha= 1,  s=350) 
scatter4 = ax3.scatter(all_data[28,0],all_data[28,3]*100,color='red', marker='*', linestyle='None', alpha= 1,  s=120)
scatter5 = ax3.scatter(all_data[8,0],all_data[8,3]*100,color='green', marker='*', linestyle='None', alpha= 1,  s=120) 

ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
plt.xlabel("Hydropower [GWh]", fontsize=14)
plt.ylabel("MEF Reliability [%]" , fontsize=14)   

    
ax3 = fig.add_subplot(426)

scatter1 = ax3.scatter(data_MEF[:,2]*100,data_MEF[:,3]*100,color='#5DC83F', alpha=0.2, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF[:,2]*100,data_noMEF[:,3]*100,color='#DC143C', alpha=0.2, linewidth=1.5)

scatter3 = ax3.scatter(data_hist_det[0,2]*100,data_hist_det[0,3]*100,color='black', alpha= 1, linewidth=1.5)
 
ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
plt.xlabel("Flood Reliability [%]", fontsize=14)
plt.ylabel("MEF Reliability [%]" , fontsize=14)   

scatter1 = ax3.scatter(data_MEF_paretosorted[:,2]*100,data_MEF_paretosorted[:,3]*100,color='#5DC83F', alpha=1, linewidth=1.5) 
scatter2 = ax3.scatter(data_noMEF_paretosorted[:,2]*100,data_noMEF_paretosorted[:,3]*100,color='#DC143C', alpha=1, linewidth=1.5)
scatter4 = ax3.scatter(all_data[28,2]*100,all_data[28,3]*100,color='black', marker='*', alpha= 1, linewidth=1.5,s = 350)
scatter5 = ax3.scatter(all_data[8,2]*100,all_data[8,3]*100,color='black', marker='*', linestyle='None', alpha= 1,s = 350)
scatter4 = ax3.scatter(all_data[28,2]*100,all_data[28,3]*100,color='red', marker='*', alpha= 1, linewidth=1.5,s = 120)
scatter5 = ax3.scatter(all_data[8,2]*100,all_data[8,3]*100,color='green', marker='*', linestyle='None', alpha= 1,s = 120)

ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
plt.xlabel("Flood Reliability [%]", fontsize=14)
plt.ylabel("MEF Reliability [%]" , fontsize=14)   

    
green_circle = mlines.Line2D([], [],c ='#5DC83F',markerfacecolor ='#5DC83F', marker='.', linestyle='None',
                          markersize=15, label='PF_MEF')


green_star = mlines.Line2D([], [],c ='None',markerfacecolor ='green', marker='*', linestyle='None',
                          markersize=15, label='Policy [PF_MEF]')

red_circle = mlines.Line2D([], [], c ='#DC143C',markerfacecolor ='#DC143C',marker='.', linestyle='None',
                          markersize=15, label='PF_nMEF')

red_star = mlines.Line2D([], [], c ='None',markerfacecolor ='red',marker='*', linestyle='None',
                          markersize=15, label='Policy [PF_nMEF]')

black_circle = mlines.Line2D([], [],c ='None',markerfacecolor ='black', marker='.', linestyle='None',
                          markersize=15, label='Historical')
    
ax3.legend([' PF_MEF ','PF_nMEF'],handles=[green_circle,red_circle, green_star,red_star,black_circle],loc='best',shadow=False, fancybox=False,bbox_to_anchor=(1.15, 1),frameon=False,fontsize=14)


#ax3.plot(x, y)
#ax3.set_title('A single plot')

#ax3.legend("PF_MEF","PF_nMEF")

plt.tight_layout()
plt.savefig('parallelplot_withsubplot.png', dpi = 1000)
