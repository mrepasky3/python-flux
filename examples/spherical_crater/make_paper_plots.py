#!/usr/bin/env python

import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

from plot_style import linewidth, dpi, marker, colors

PAPER_PLOT_DIR = sys.argv[1]
SAVE_PDF_PLOTS = False

def load_stats(path):
    with open(path, 'r') as f:
        return json.load(f)

def print_stats(stats):
    print(json.dumps(stats, sort_keys=True, indent=4))

def read_all_stats_files_to_dicts(glob_path, verbose=False):
    Stats = dict()
    for test_path in glob.glob(glob_path):
        p = int(test_path.split('_')[-1][1:])
        stats_path = os.path.join(test_path, 'stats.json')
        stats = load_stats(stats_path)
        if verbose:
            print_stats(stats)
        Stats[p] = stats
    return Stats

def tol_from_path(path):
    return path.split('_')[-1]

def tol_to_tex(tol):
    return '10^{%d}' % int(tol[2:])

def load_direct_comparison_data_to_dict(path):
    Data = dict()
    with open(path/'B_rel_l2_errors.pickle', 'rb') as f:
        Data['B_rel_l2_errors'] = pickle.load(f)
    with open(path/'T_rel_l2_errors.pickle', 'rb') as f:
        Data['T_rel_l2_errors'] = pickle.load(f)
    with open(path/'FF_rel_fro_errors.pickle', 'rb') as f:
        Data['FF_rel_fro_errors'] = pickle.load(f)
    return Data

def get_values_by_key(dicts, key):
    lst = []
    for k in sorted(dicts):
        d = dicts[k]
        lst.append(d[key])
    return lst

stats_gt_path = 'stats/gt'
stats_path_pattern = 'stats/eps_*'
comparison_path_pattern = 'stats/gt_vs_*'

################################################################################
# MAKE INGERSOLL PLOTS
#

# plot parameters

matplotlib.rcParams.update({'font.size': 18})

# Load statistics
StatsGt = read_all_stats_files_to_dicts(os.path.join(stats_gt_path, 'ingersoll_p*'))
Stats = {
    tol_from_path(path): read_all_stats_files_to_dicts(
        os.path.join(path, 'ingersoll_p*'))
    for path in glob.glob(stats_path_pattern)
}

# Load direct comparison data
GtVsTol = {
    tol_from_path(path): load_direct_comparison_data_to_dict(
        Path(f'./stats/gt_vs_{tol_from_path(path)}'))
    for path in glob.glob(comparison_path_pattern)
}

Tols = list(GtVsTol.keys())

# Get values of H
H = get_values_by_key(StatsGt, 'h')
N = get_values_by_key(StatsGt, 'num_faces')

# Make loglog h vs T_rms plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 'rms_error'),
           linewidth=linewidth, marker='o', c=colors[0], label='Sparse $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(H, get_values_by_key(Stats[tol], 'rms_error'),
               linewidth=linewidth, marker=marker, c=colors[i + 1],
               linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('RMS error in $T$ (shadow)')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_rms.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_rms.png', dpi=dpi)
plt.close()

# Make loglog h vs T_rel_l2_errors and B_rel_l2_errors plot
plt.figure(figsize=(6, 6))
for i, tol in enumerate(Tols):
    plt.loglog(
        H, GtVsTol[tol]['T_rel_l2_errors'],
        linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='-',
        label=r'$\|T_{gt} - T\|_2/\|T_{gt}\|_2$ ($\epsilon = %s$)' % (
            tol_to_tex(tol),))
    plt.loglog(
        H, GtVsTol[tol]['B_rel_l2_errors'],
        linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
        label=r'$\|B_{gt} - B\|_2/\|B_{gt}\|_2$ ($\epsilon = %s$)' % (
            tol_to_tex(tol),))
plt.legend()
plt.xlabel('$h$')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_ptwise_errors.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_ptwise_errors.png', dpi=dpi)
plt.close()

# Make loglog h vs FF_rel_fro_errors
plt.figure(figsize=(6, 6))
for i, tol in enumerate(Tols):
    plt.loglog(H, GtVsTol[tol]['FF_rel_fro_errors'],
               linewidth=linewidth, marker=marker, c=colors[i + 1],
               linestyle='-',
               label=r'$\epsilon = %s$' % (tol_to_tex(tol),))
plt.legend()
plt.ylabel(r'$\|\mathbf{F}_{gt} - \mathbf{F}\|_{F}/\|\mathbf{F}_{gt}\|_{F}$')
plt.xlabel(r'$h$')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/FF_rel_fro_errors.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/FF_rel_fro_errors.png', dpi=dpi)
plt.close()

# Make loglog h vs size plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 'FF_size'),
           linewidth=linewidth, marker=marker, c=colors[0], label='Sparse $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(H, get_values_by_key(Stats[tol], 'FF_size'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Size of $F$ [MB]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_size.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_size.png', dpi=dpi)
plt.close()

# Make loglog h vs compute T time plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 't_T'),
           linewidth=linewidth, marker=marker, c=colors[0], label='Sparse $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(H, get_values_by_key(Stats[tol], 't_T'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Time to compute $T$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_T_time.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_T_time.png', dpi=dpi)
plt.close()

# Make loglog h vs compute B and E time plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 't_B'),
           linewidth=linewidth, marker=marker, c=colors[0], label='Sparse $F$', zorder=1)
plt.loglog(H, get_values_by_key(StatsGt, 't_E'), linewidth=linewidth, marker=marker,
           c=colors[0], linestyle='--',
           label='Compute $E$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(get_values_by_key(Stats[tol], 'h'),
               get_values_by_key(Stats[tol], 't_B'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Time to compute $B$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_B_and_E_time.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_B_and_E_time.png', dpi=dpi)
plt.close()

# Make loglog h vs assembly time plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 't_FF'),
           linewidth=linewidth, marker=marker, c=colors[0], label='Sparse $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(H, get_values_by_key(Stats[tol], 't_FF'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Time to assemble $F$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_assembly_time.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/h_vs_assembly_time.png', dpi=dpi)
plt.close()