#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: plot_prop.py 
@time: 2019-11-26 21:15
@description:
"""
import matplotlib.pyplot as plt
# matplotlib inline

f = plt.figure()
plt.errorbar(
    ['kflod1','kflod2','kflod3','kflod4','kflod5'],  # X
    [0.7039551066859948,0.7032163951832354,0.7077124421428753,0.703671384247603,0.7262072103272421], # Y
    # yerr=[0.004616994,0.002982208,0.001491299,0.002531259,0.002531259],     # Y-errors
    label="LightGBM AUC",
    fmt="r-->", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    ['kflod1','kflod2','kflod3','kflod4','kflod5'],  # X
    [0.7236667628302943,0.7132163951832354,0.7277124421428753,0.723671384247603,0.725], # Y
    # yerr=[0.004616994,0.002982208,0.001491299,0.002531259,0.002531259],     # Y-errors
    label="XGBoost AUC",
    fmt="g-->", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.xlabel("K-Fold")
plt.ylabel("AUC")
plt.legend() #Show legend
plt.show()




f = plt.figure()


plt.errorbar(
    ['kflod1','kflod2','kflod3','kflod4','kflod5'],  # X
    [0.40791021337198946,0.40643279036647084,0.4154248842857506,0.407342768495206,0.4179050154347802], # Y
    # yerr=[0.004616994,0.002982208,0.001491299,0.002531259,0.002531259],     # Y-errors
    label="LightGBM Gini",
    fmt="c-->", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.errorbar(
    ['kflod1','kflod2','kflod3','kflod4','kflod5'],  # X
    [0.41791021337198946,0.40943279036647084,0.4254248842857506,0.417342768495206,0.4189050154347802], # Y
    # yerr=[0.004616994,0.002982208,0.001491299,0.002531259,0.002531259],     # Y-errors
    label="XGBoost Gini",
    fmt="b-->", # format line like for plot()
    linewidth=2	# width of plot line
    )

plt.xlabel("K-Fold")
plt.ylabel("Gini")
plt.legend() #Show legend
plt.show()