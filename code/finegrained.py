#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:20:26 2018

@author: rgugg
"""
import sys, os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
sys.path.append(os.path.split(__file__)[0])
from simulate_dependency import Simulation

# %%
def get_even_bands():
    bandlist = []    
    for bandwidth in reversed(range(1, 16, 1)):  
        band = (frequency-bandwidth, frequency+bandwidth)
        bandlist.append(band) 
    return bandlist

def get_uneven_bands():
    bandlist = []    
    for bandwidth in reversed(range(1, 16, 1)):  
        band = (frequency-.5*bandwidth, frequency+1.5*bandwidth)
        bandlist.append(band) 
    return bandlist
    
def collect(sim, bandlist=None):
    bcoll = []
    yticklabels = []
    
    if bandlist is None:
        bandlist = get_even_bands()

    for band in bandlist:                
        yticklabels.append('{0:2.0f}:{1:2.0f}'.format(*band))
        coll = []
        for oidx, order in enumerate(range(0, 9, 1)):  
            sim.generate(0)
            sim.butterfilter(bandLimits=band, filterOrder=order)
            p, a = sim.pick()
            coll.append(p)
        bcoll.append(coll)
    coll = np.asanyarray(bcoll)
    return coll, yticklabels

# %%
frequency = 18
cmap = sns.color_palette("RdBu_r", 72)

sim = Simulation(mode=(3, 1.5, 1000), 
                 dependency='rising', 
                 frequency=frequency, 
                 noiselevel=0)

projectFolder = './'
resultsFolder = projectFolder + 'results/overview/'
os.makedirs(resultsFolder,  exist_ok=True)
#%%
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
sim.pick = sim._pick_fourier
coll, ylabs = collect(sim)
tmp = np.rad2deg(coll)
ax = sns.heatmap(tmp, ax=ax, cmap=cmap, cbar=False, center=-90)
ax.set_yticklabels(ylabs, rotation=0)
ax.set_title('Fourier')
ax.set_ylabel('Bandwidth')
ax.set_xlabel('Filter Order')
fig.tight_layout()
fig.savefig(resultsFolder + 'fourier_even.png')

sim.pick = sim._pick_hilbert
coll, ylabs = collect(sim)
tmp = np.rad2deg(coll)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
ax = sns.heatmap(tmp, ax=ax, cmap=cmap, cbar=False, center=-90)
ax.set_yticklabels(ylabs, rotation=0)
ax.set_ylabel('Bandwidth')
ax.set_title('Hilbert')
ax.set_xlabel('Filter Order')
fig.tight_layout()
fig.savefig(resultsFolder + 'hilbert_even.png')
# %% 
bandlist = get_uneven_bands()

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
sim.pick = sim._pick_fourier
coll, ylabs = collect(sim, bandlist=bandlist)
tmp = np.rad2deg(coll)
ax = sns.heatmap(tmp, ax=ax, cmap=cmap, cbar=False, center=-90)
ax.set_yticklabels(ylabs, rotation=0)
ax.set_title('Fourier')
ax.set_ylabel('Bandwidth')
ax.set_xlabel('Filter Order')
fig.tight_layout()
fig.savefig(resultsFolder + 'fourier_uneven.png')


sim.pick = sim._pick_hilbert
coll, ylabs = collect(sim,  bandlist=bandlist)
tmp = np.rad2deg(coll)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
ax = sns.heatmap(tmp, ax=ax, cmap=cmap, cbar=True, center=-90)
ax.set_yticklabels(ylabs, rotation=0)
ax.set_ylabel('Bandwidth')
ax.set_title('Hilbert')
ax.set_xlabel('Filter Order')
fig.tight_layout()
fig.savefig(resultsFolder + 'hilbert_uneven.png')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
ax = sns.heatmap(tmp, ax=ax, cmap=cmap, cbar=True, center=-90)
fig.savefig(resultsFolder + 'cbar.png')