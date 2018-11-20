#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:26:35 2017

@author: rgugg

According to Keil et al. (2013):
The EEG and EOG signals were sampled at 2,000 Hz with an online low-pass filter of 200 Hz.
Epochs of 3 s around the TMS pulse were extracted from the raw data of 
the motor involuntary condition. A linear trend was removed from each epoch, 
and power line noise was removed by rejecting the 60-Hz bin from the 
epoch's spectrum using a discrete Fournier transform. Resulting epochs 
were inspected for artefacts, and channels with excessive noise or flat 
lines were interpolated. [...] The EEG and EMG signal was band-pass filtered for the frequency
of interest (17–19 Hz, 8th-order Butterworth filter, one pass). To extract
power and phase angles, a Hilbert transform was computed on three cycles 
of the 18-Hz frequency of interest prior to the upramp of the TMS artefact.
Power values were computed from the absolute of the Hilbert transformed 
signal. We could not determine the exact phase at which the pulse arrived, 
because the upramp of the TMS artefact required us to insert a small delay 
(5 ms) between the extracted phase and the recorded TMS pulse.
'''

''' van Elswijk (2010)
EMG [...] were acquired using standard procedures
 (10,000 Hz). Electroencephalogram (EEG) was [...] acquired using standard 

procedures (2000 Hz). [...] The raw EMG signal was cut into epochs of ±1.1 s around the TMS pulse.
These epochs contained a small TMS artifact that was restricted to the first 
1.5 ms (15 samples) after the TMS pulse. The EMG signal was bandpass filtered 
between 10 and 400 Hz (fourth-order Butterworth). Filtering was performed only
forward in time, i.e., causal, to prevent any post-TMS effect from leaking 
into pre-TMS time. [...]
For each frequency, we used an epoch that had a length of two cycles at that
frequency and that ended with the TMS pulse. This epoch was multiplied with
a Hanning taper and Fourier transformed to give the phase and amplitude at 
the respective frequency. 
"""
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack as fft
import numpy as np
# %% simulation
class Simulation():
    '''Class implements a EEG-MEP simulation according to Keil et al. 2013,
    van Elswijk et al. 2010 or specified on arbitrary parameters
        
    example
    -------
    sim = Simulation(mode = 'keil', frequency = 18, dependency = 'rising')
    sim.generate(startPhase = 17)
    sim.butterfilter(bandLimits = (17, 19), filterOrder=4)
    eeg, mep = sim.pick()
    '''
    
    
    amplitudeEEG = 1/np.sqrt(2) #: scaling factor to normalize EEG amplitude
    averageMep = 50 #: scaling factor as base for log-normal MEP amplitude
    
    def __init__(self,mode = 'keil',                                 
                    dependency = 'rising',
                    frequency = 18, 
                    noiselevel = 0,
                    pick=None):
        '''
        args
        ----
        mode: str or 3-tuple
            - can be 'elswijk' for cycles = 2, duration  = +-1.1 and fs = 10000
            - can be 'keil' for cycles = 3, duration  =  +-1.5 and fs = 2000
            - can be a tuple (cycles:int, duration:float, fs: int)
        
        dependency: str
            can be 'rising' for sinusoidal modulation exhibting highest MEP at 
            the rising flank or
            'bimodal' for highest MEPs at peak and trough of the EEG

        frequency:float
            frequency at which the EEG-phase determines MEP amplitude
        
        noiselevel:float
            standard deviance of the white noise added to EEG and MEP simulation 
            
        t0: int
            Index to pick the EEG cycles for phase estimation and amplitude for 
            MEP modulation. Defaults to None, which will be overwritten as the 
            index of the center of the EEG sample -> considering +- duration
        ''' 
        if mode == 'elswijk':
            cycles = 2
            duration = 1.1
            fs = 10000
            bias = 0
            if pick is None or pick == 'fourier':
                self.pick = self._pick_fourier            
            else:
                raise ValueError('Elswijk must use Fourier')
        elif mode == 'keil':
            cycles = 3
            duration = 1.5
            fs = 2000
            bias = int(0.05*fs)          
            if pick is None or pick == 'hilbert':
                self.pick = self._pick_hilbert            
            else:
                raise ValueError('Keil must use Hilbert')
        elif type(mode) is tuple:
            cycles, duration, fs = mode
            bias = 0
            if pick is None or pick == 'hilbert':
                self.pick = self._pick_hilbert
            else:
                self.pick = self._pick_fourier
        else:
            raise NotImplementedError       
        
        self.frequency= frequency
        self.fs = fs
        self.bias = bias
        self.duration = 2 * duration
        self.cycles = cycles
        self.noiselevel = noiselevel        

        self.set_picker()
        self.set_sampling()        
        self.set_generator(dependency);
    
    def set_sampling(self):
        'sets the parameters of the generator based on self.mode'
        self.samples = int(self.duration*self.fs)+1                
        self.T = np.linspace(0, self.duration, self.samples)        
    
    def set_picker(self):
        'sets the picker based on the self.mode'
        #timepoint of TMS pulse            
        self.t0 = int((self.duration*self.fs)//2)            
        #start is n cycles before TMS
        start = -int(self.cycles * self.fs / self.frequency) 
        # if there is a offset from TMS, start earlier
        start = start-self.bias 
        # toi is from start to bias centered on t0
        self.toi = self.t0 + np.arange(start, -self.bias, 1)                    

    def set_generator(self, dependency):
        'sets the generator based on the dependency '
        self.dependency = dependency
        if dependency is 'rising':
            self._gen_dependent = self._gen_rising
        elif dependency is 'bimodal':
            self._gen_dependent = self._gen_bimodal     
            
    def _gen_rising(self, X):
        'implements maximum at the rising flank of the eeg'
        mepScale = (1 + np.cos(X))/2
        return mepScale
    
    def _gen_bimodal(self, X):
        ''' bimodal maxima at peak and trough of the eeg'
            
            args
            ----
            X:ndarray
                normalized time units
        '''

        shift = 2 * np.pi * (90/360) * self.frequency
        mepScale = (1+np.cos(2 * X + shift))/2
        return mepScale
                
    def whitenoise(self):
        'generate white noise with size equal to self.samples'
        return np.random.normal(loc=0.0, scale=self.noiselevel, 
                               size=self.samples) 
    
    def generate(self, startPhase = 0):      
        '''Generate a new simulation of eeg and mep
        
        args
        ----
        startPhase: float
            the starting phase of the eeg signal
            
        return
        ------
        eeg: numpy.ndarray
            stores the EEG simulation in self.eeg
        mep: numpy.ndarray
            stores the MEP simulation in self.mep
        '''
        phaseShift = 2 * np.pi * (startPhase/360)
        X = (2 * np.pi * self.frequency * self.T) + phaseShift
        mepScale = self._gen_dependent(X)
        
        self.mep = self.averageMep**(mepScale + self.whitenoise())
        self.eeg = self.amplitudeEEG * np.sin(X)  + self.whitenoise()    


    def butterfilter(self, bandLimits = (17,19), filterOrder = 8):
        '''Apply causal butterworth filter to self.eeg
        
        args
        ----
        bandlimits: tuple
            specifiying (min, max) of the passband
        
        filterOrder: int
            specifying the order of the butterworth filter
    
        return
        ------    
        filtered:numpy.ndarray
            stores the results of the filter in self.filtered
    
        '''        
        nyquist = self.fs/2
        band = (bandLimits[0] / nyquist, bandLimits[1] / nyquist )
        b, a = signal.butter(filterOrder, band, 
                             btype = 'bandpass',
                             analog = False)
        self.filtered = signal.lfilter(b, a, self.eeg)
    
    
    def _pick_hilbert(self):
        'returns mep at t0 and angle based on Hilbert Transformation'
        tmp = self.filtered[self.toi]
        angle = np.angle(signal.hilbert(tmp))   
        mep = self.mep[self.t0]
        return angle[-1], mep        
    
    def _pick_fourier(self):
        'returns mep at t0 and angle based on Fourier Transformation'
        # based on length of segment, which integer natural frequency is
        # closest to the frequency of interest. Add 1 due to first entry being DC.        
        # As the segment has a length of integer cycles -> 
        # indexing starts at zero, cycles at 1, therefore no +1 necessary
        foi = self.cycles
        tmp = self.filtered[self.toi]
        win = np.hanning(tmp.shape[0])        
        angle = np.angle(fft.fft(tmp*win))
        mep = self.mep[self.t0]
        return angle[foi], mep
        
        
# %% Visualization
def normalize_mep(mep):  
    'Normalizes a list of mep values by their maximum'
    tmp = np.asarray(mep)
    mep = tmp/np.max(tmp)
    return mep

def collect_simulations(sim, bl, fo):
    A, M = [], []
    for phase in range(0, 360, 1):     
        sim.generate(startPhase = phase)
        if fo >= 0:
            sim.butterfilter(bandLimits = bl, filterOrder=fo)
        else:
            sim.filtered = sim.eeg    
        angle, mep = sim.pick()                
        A.append(angle)
        M.append(mep)
    return A, normalize_mep(M)
            
def order(n):
    part = str(n)+("th" if 4<=n%100<=20 else {1:"st",2:"nd",3:"rd"}.get(n%10, "th"))
    return part + ' order'

def tableau(items = ['o','s']):        
    idx = 0
    while idx < len(items):
        yield items[idx]
        idx += 1
        idx = idx % len(items)

def scatterplot(ax,angle, mep, annotation, mcolor = 'k', mtype = 'o'):
    'plots angle and mep for a linear-linear scatterplot'            
    ax.scatter(angle, mep,
               alpha=1,               
               edgecolors = 'gray',
               linewidths = 0.5,
               facecolors=mcolor,
               marker=mtype,
               )     
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels(["-\u03C0","-\u03C0/2","0", "\u03C0/2","\u03C0" ], fontsize = 18) 
    ax.set_title(annotation, fontsize = 20)        
    
    
def polarplot(ax, angle, mep, annotation, mcolor = 'k', mtype = 'o'):
    'plots angle and mep for a polar-linear scatterplot'     
    ax.scatter(angle, mep, 
               alpha=1,               
               edgecolors='gray',
               linewidths=0.5,
               facecolors=mcolor,
               marker=mtype,
               )
    ax.set_title(f'{annotation}', va='baseline', loc = 'left')
    ax.set_yticklabels('')
    ax.set_yticks(np.arange(0,1.1, .1))
    ax.set_ylim(0, 1.1)
    ax.title.set_fontsize(14)
#%%
if __name__ == '__main__':
    # %% project configuration
    import os    
    projectFolder = './'
    resultsFolder = projectFolder + 'results/'
    os.makedirs(resultsFolder,  exist_ok=True)
    depmodes= ('rising', 'bimodal')
    studymodes = ['keil','elswijk']
    blmodes = [(10, 400),(17, 19)]
    # %% Visualize Dependency    
    plt.close('all')
    fig, axes = plt.subplots(2,1, figsize = (9,6), dpi = 300,sharex = True)
    fig.subplots_adjust(hspace=0.3)
    
    for ax, (idx, dep) in zip(axes, enumerate(depmodes)):   
        sim = Simulation(mode = (3, 1.5/18, 5000), dependency = dep)        
        sim.generate(startPhase = 0)
        
        ax.plot(normalize_mep(sim.mep),color='k')
        ax.plot(sim.eeg,color = '.5')
        ax.set_title(dep.capitalize())
        ax.set_xticks(np.linspace(0,sim.samples-1,4))
        ax.set_xticks(np.linspace(0,sim.samples-1,13), minor = True)        
        ax.set_xticklabels([0,1,2,3])    
        ax.set_yticks(np.linspace(-1,1,3))    
        ax.set_yticks(np.linspace(-1,1,9), minor = True)    
        ax.set_yticklabels([-1,0,1])    
        ax.set_ylim(-1,1.1)
        ax.grid(which = 'both')
        ax.grid(which='minor', alpha=0.25)                                                
        ax.grid(which='major', alpha=0.75) 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.legend(('MEP', 'Oscillation'))     
        
    ax.set_xlabel('Time in periods of 2 \u03C0')
    fig.text(0.075, 0.5, 'Normalized Amplitude', va='center', rotation='vertical')
    fig.savefig(resultsFolder + 'trace_sim.png')
    fig.savefig(resultsFolder + 'trace_sim.eps')
    # %% Run All Simulations    
    filter_range = np.arange(0,9,1)
    plt.close('all')
    for dep in depmodes:
        for study in studymodes:
            sim = Simulation(mode = study, dependency = dep, noiselevel = 0.05)
            for bl in blmodes:
                fig1, axes1 = plt.subplots(1, len(filter_range), 
                                           figsize = (27,6), dpi = 300,
                                           subplot_kw={'projection':'polar'})
                fig1.subplots_adjust(wspace = 0.5)
                fig2, axes2 = plt.subplots(1, len(filter_range), 
                                           figsize = (27,6), dpi = 300,
                                           sharex = True, sharey = True)
                fig2.subplots_adjust(wspace = 0.5)
                for ax1, ax2, fo in zip(axes1.flatten(), axes2.flatten(),
                                        filter_range):
                    A, M = collect_simulations(sim, bl, fo)                       
                    polarplot(ax1, A, M, order(fo))
                    scatterplot(ax2, A, M, order(fo))
                    
                plt.tight_layout()
                os.makedirs(resultsFolder + '/' + study, exist_ok=True)
                fig1.savefig(resultsFolder + '/' + study + '/polar_' + dep +  
                            f'_{bl[0]}_to_{bl[1]}' + '.png')   
                
                fig2.savefig(resultsFolder + '/' + study + '/scatter_' + dep + 
                            f'_{bl[0]}_to_{bl[1]}' + '.png')  
                fig1.savefig(resultsFolder + '/' + study + '/polar_' + dep +  
                            f'_{bl[0]}_to_{bl[1]}' + '.eps')   
                
                fig2.savefig(resultsFolder + '/' + study + '/scatter_' + dep + 
                            f'_{bl[0]}_to_{bl[1]}' + '.eps')   
    plt.close('all')
    # %%  Replicate  & vice versa
    plt.close('all')
        
    fig = plt.figure(figsize = (10,5), dpi = 300)
    ax1 = fig.add_subplot(1,2,1)    
    ax1.set_ylabel('Normalized Amplitude', fontsize= 12)
    ax1.grid()
    ax2 = fig.add_subplot(1,2,2, projection = 'polar')
    sim = Simulation(mode = 'keil', dependency = 'bimodal', noiselevel = 0.05)
    A, M = collect_simulations(sim, (17, 19), 8)
    scatterplot(ax1, A, M,'', '.8','o')    
    polarplot(ax2, A, M,'', '.8','o')
    sim = Simulation(mode = 'elswijk', dependency = 'bimodal', noiselevel = 0.05)
    A, M = collect_simulations(sim, (10, 400), 4)
    scatterplot(ax1, A, M, '', '.5','s')
    polarplot(ax2, A, M, '','.5','s')
    ax1.legend(['Keil','van Elswijk'],edgecolor= 'w', loc = (1.05,0.95), framealpha = 1)    
    ax1.set_title('Bimodal', fontsize= 16)
    fig.savefig(resultsFolder + 'recover_bimodal.png')   
    fig.savefig(resultsFolder + 'recover_bimodal.eps')  

    fig = plt.figure(figsize = (10,5), dpi = 300)
    ax1 = fig.add_subplot(1,2,1)    
    ax1.set_ylabel('Normalized Amplitude', fontsize= 12)    
    ax1.grid()
    ax2 = fig.add_subplot(1,2,2, projection = 'polar')
    sim = Simulation(mode = 'keil', dependency = 'rising', noiselevel = 0.05)
    A, M = collect_simulations(sim, (17, 19), 8)
    scatterplot(ax1, A, M,'', '.8','o')    
    polarplot(ax2, A, M,'', '.8','o')
    sim = Simulation(mode = 'elswijk', dependency = 'rising', noiselevel = 0.05)
    A, M = collect_simulations(sim, (10, 400), 4)
    scatterplot(ax1, A, M, '', '.5','s')
    polarplot(ax2, A, M, '','.5','s')
    ax1.set_title('Rising', fontsize= 16)
    fig.savefig(resultsFolder + 'recover_rising.png')   
    fig.savefig(resultsFolder + 'recover_rising.eps')   
