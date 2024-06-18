from typing import List, Dict
import os
import subprocess
import numpy as np

from src.utils import *
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load simulation results

# RWDB = f'PFGSE_NMR_18102023'
# SIMS_DIR = f'db/{RWDB}/carbonates'
# SIM_DIR = [d for d in os.listdir(SIMS_DIR) if parse_dirname(d)['sample'] == SAMPLE][0]

def sim_data(SIMS_DIR, rock_type):
    SIM_DIR = [d for d in os.listdir(SIMS_DIR) if parse_dirname(d)['sample'] == rock_type][0]
    SIM_DIRPATH = os.path.join(SIMS_DIR, SIM_DIR)
    sim_info = parse_dirname(SIM_DIR)
    pfg_dir = [d for d in os.listdir(SIM_DIRPATH) if 'NMR_pfgse' in d]
    timesamples_dir = os.path.join(SIM_DIRPATH, pfg_dir[0], 'timesamples')

    results_file = os.path.join(SIM_DIRPATH, pfg_dir[0], 'PFGSE_results.csv')
    gradients_file = os.path.join(SIM_DIRPATH, pfg_dir[0], 'PFGSE_gradient.csv')
    echoes_files = [os.path.join(timesamples_dir, f) for f in os.listdir(timesamples_dir) if 'echoes' in f]
    echoes_files = order_files_by_last_token(echoes_files)

    sim_results_data = parse_sim_results(results_file, ['Time','Dmsd','Dsat', 'Dsat(pts)'])
    sim_gradients_data = parse_sim_results(gradients_file, ['Gz', 'Kz'])
    sim_echoes_data = [parse_sim_results(ef, ['Gradient','NMR_signal(mean)']) for ef in echoes_files]

    sim_data = {
        'info': sim_info,
        'results': sim_results_data,
        'gradients': sim_gradients_data,
        'echoes': sim_echoes_data
    }

    return sim_data

def sort_list_based_on_another_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

def sphere_dir_organizer(dirSph, rock_type):
    
    SPHERES_DB = dirSph
    SPHERES_DIRS = [os.path.join(SPHERES_DB, d) for d in os.listdir(SPHERES_DB) if rock_type in parse_dirname(d)['sample']]

    tokens = [
        sd
        .split('/')[-1]
        .split('_')[2]
        .split('sphere')[-1] 
        for sd in SPHERES_DIRS
    ]
    tokens = np.array([int(t) for t in tokens])
    tokens



    return sort_list_based_on_another_list(SPHERES_DIRS, tokens)

def sphere_data(dirSph, rock_type):
    spheres_data = []

    for sd in sphere_dir_organizer(dirSph, rock_type):
        sd_info = parse_dirname(sd)
        sd_info['sample'] = sd_info['sample'].split('=')[-1]
        sd_info['radius'] = float(sd_info['sample'].split('sphere')[-1])

        pfg_dir = [os.path.join(sd,d) for d in os.listdir(sd) if 'NMR_pfgse' in d]
        timesamples_dir = os.path.join(pfg_dir[0], 'timesamples')

        sd_results_file = os.path.join(pfg_dir[0], 'PFGSE_results.csv')
        sd_gradients_file = os.path.join(pfg_dir[0], 'PFGSE_gradient.csv')
        sd_echoes_files = [os.path.join(timesamples_dir, f) for f in os.listdir(timesamples_dir) if 'echoes' in f]
        sd_echoes_files = order_files_by_last_token(sd_echoes_files)

        sd_results_data = parse_sim_results(sd_results_file, ['Time','Dmsd','Dsat', 'Dsat(pts)'])
        sd_gradients_data = parse_sim_results(sd_gradients_file, ['Gz', 'Kz'])
        sd_echoes_data = [parse_sim_results(ef, ['Gradient','NMR_signal(mean)']) for ef in sd_echoes_files]

        sd_data = {
            'info': sd_info,
            'results': sd_results_data,
            'gradients': sd_gradients_data,
            'echoes': sd_echoes_data
        }    
        spheres_data.append(sd_data)
    return sphere_data

#scriptW


def data_combining(sim_data, spheres_data, sim_fraction):
    

        
    combined_data = []

    for sphere in spheres_data:
        new_sample = {}
        new_sample['sample'] = f"{sim_data['info']['sample']}_with_{sphere['info']['sample'][2:]}"
        new_sample['radius'] = sphere['info']['radius']
        new_sample['Kz'] = sim_data['gradients']['Kz']
        new_sample['echoes'] = []
        new_sample['time'] = []

        for ie, echo in enumerate(sphere['echoes']):
            new_signal = sim_fraction * sim_data['echoes'][ie]['NMR_signal(mean)']
            new_signal += (1-sim_fraction) * echo['NMR_signal(mean)']
            new_sample['echoes'].append(new_signal)
            new_sample['time'].append(sim_data['results']['Time'][ie])
        combined_data.append(new_sample)
    return combined_data
    
    
    
def fit_pfg_diffusion(Mkt, k, time, delta, threshold=0.9, cutoff=1, min_values=2):
    
    xdata = k * k
    xdata = (-1) * (time - delta/3.0) * xdata
        
    M0 = Mkt[0]
    ydata = (1.0/M0) * Mkt
    
    cutoff = find_cutoff_point(ydata, threshold, cutoff, min_values)
    
    ydata = np.log(ydata)
    Dt_fit = linear_regression_numpy(xdata[:cutoff], ydata[:cutoff], fit_intercept=False)
    return Dt_fit[0]

def compute_dT(combined_data, SMALL_DELTA):
    for cd in combined_data:
        cd['Dsat'] = []
        for i,e in enumerate(cd['echoes']):
            Mkt = e
            Kz = cd['Kz']
            time = cd['time'][i]     
            Dt_fit = fit_pfg_diffusion(Mkt, Kz, time, SMALL_DELTA, threshold=0.9, min_values=25)
            cd['Dsat'].append(Dt_fit)
        cd['Dsat'] = np.array(cd['Dsat'])
    return combined_data



def plot(rock_type, spheres_data, combined_data, exp_data, sim_fraction, D0):
    fig, axs = plt.subplots(2,2,figsize=(9,6.5),constrained_layout=True)
    stride = 2

    suptitle = TAG2SAMPLE_MAP[rock_type]
    suptitle += f"\nPorosities: {100*sim_fraction:.1f}% simulation + {100*(1-sim_fraction)['sphere']:.1f}% isolated sphere"
    fig.suptitle(suptitle)

    for i, ax in enumerate(axs.flatten()):
        curr_rad = f"{float(spheres_data[i]['info']['res'])*spheres_data[i]['info']['radius']:.0f}"
        title = r"$r=$" + curr_rad + r" $\mu m$"
        ax.set_title(title, x=0.3, y=1, pad=-18)
        # plot experimental data

        ax.scatter(
            exp_data['data']['time'], 
            exp_data['data']['D_D0'],
            marker='s',
            color='blue',
            label='experiment'
        )

        # plot simulation data
        ax.scatter(
            sim_data['results']['Time'][::stride], 
            (1/D0)*sim_data['results']['Dsat'][::stride],
            marker='o',
            color='red',
            label='simulation'
        )

        # plot isolated spheres data
        cd = combined_data[i]
        ax.scatter(
            cd['time'][::stride], 
            (1/D0)*cd['Dsat'][::stride],
            marker='*',
            facecolor='none',
            color='gold',
            label=f"sim + sphere",
        )

    for ax in axs.flatten():
        ax.set_xlim([0,1.1*exp_data['data']['time'].iloc[-1]])
        ax.set_ylim([0,1])
        ax.set_xlabel('time [msec]')
        ax.set_ylabel(r'$D(t)/D_0$')
        ax.legend(loc='upper right', frameon=False)
        ax.label_outer()


def eda_vuggys(rock_type, dirSph, SIMS_DIR, exp_data, sim_fraction, D0):
    sphere = sphere_data(dirSph, rock_type)
    sim = sim_data(SIMS_DIR, rock_type)
    combined_data = data_combining(sim, sphere, sim_fraction)
    plot(rock_type, sphere_data, combined_data, exp_data, sim_fraction, D0)


