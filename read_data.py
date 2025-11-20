# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:14:28 2024

@author: alec
"""

from matplotlib import pyplot as plt
from boutdata import collect

path = "D:/test/T_100"

n = collect('n', path = path)
n = n.squeeze()

#%% plot tætheden i sidste tidsskridt

fig, ax = plt.subplots()
ax.contourf(n[-1, :, :].T)
