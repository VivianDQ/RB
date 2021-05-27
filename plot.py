import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab
import pickle

def attack_name(attack):
    name = {
        'garcelon_strongest': '$\\bf{Garcelon\ topN\ (N=K-1)}$',
        'garcelon_topN': '$\\bf{Garcelon\ topN\ (N=0.5K)}$',
        'oracle_strongest': '$\\bf{Oracle\ topN\ (N=K-1)}$',
        'oracle_topN': '$\\bf{Oracle\ topN\ (N=0.5K)}}$',
        'flip_theta': '$\\bf{Flip\ theta}$',
        'context_strongest': '$\\bf{Context\ topN\ (N=K-1)}$',
        'context_topN': '$\\bf{Context\ topN\ (N=0.5K)}$',
    }
    return name[attack]

def attack_figure_name(attack):
    name = {
        'garcelon_strongest': 'Garcelon\ topN\ (N=K-1)',
        'garcelon_topN': 'Garcelon\ topN\ (N=0.5K)',
        'oracle_strongest': 'Oracle\ topN\ (N=K-1)',
        'oracle_topN': 'Oracle\ topN\ (N=0.5K)',
        'flip_theta': 'Flip\ theta',
        'context_strongest': 'Context\ topN\ (N=K-1)',
        'context_topN': 'Context\ topN\ (N=0.5K)',
    }
    return name[attack]

def draw_C():
    plot_style = {
            'linucb': ['-.', 'green', '$\\bf{LinUCB}$'],
            'lints': [':', 'purple', '$\\bf{LinTS}$'],
            'rb': ['-', 'red', '$\\bf{RobustBandit}$'],
            'greedy': ['--', 'blue', '$\\bf{Greedy}$'],
        }
    plot_prior = {
            'linucb': 1,
            'lints': 2,
            'greedy': 3,
            'rb': 4,
        }
    root = 'results/overc/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    cat = os.listdir(root)
    paths = []
    for c in cat:
        paths.append(root + c)
    for path in paths:
        attack = path.split('/')[-1][:-4]
        plot_name =  'overc_' + attack
        title = '{}'.format(attack_name(attack))
        print('plotting overc in {}, output in {}.pdf'.format(path, plot_name))
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 20, 'axes.labelsize': 18, 'font.size': 20, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8), 'figure.max_open_warning': 0}
        pylab.rcParams.update(params)
        leg = []
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        Cs = [0, 10**2, 10**3, 10**4, 10**5]
        keys = sorted(plot_style.keys(), key=lambda kv: plot_prior[kv])
        for key in keys:
            if key not in data.keys(): continue
            leg += [plot_style[key][-1]]
            plot.plot(
                list(range(len(Cs))), 
                data[key], 'k', linestyle = plot_style[key][0], 
                color = plot_style[key][1], linewidth = 2
            )
            plot.legend((leg), loc='upper left', fontsize=12, frameon=False)
        dates = Cs
        plot.xticks(list(range(len(Cs))), dates)
        plot.xlabel('Attack Budget')
        plot.ylabel('Cumulative Regret')
        plot.title(title)
        fig.savefig('plots/' + plot_name + '.pdf', dpi=300, bbox_inches = "tight")
        
def draw_figure(T = 10**6):
    plot_style = {
            'linucb': ['-.', 'green', '$\\bf{LinUCB}$'],
            'lints': [':', 'purple', '$\\bf{LinTS}$'],
            'rb': ['-', 'red', '$\\bf{RobustBandit}$'],
            'greedy': ['--', 'blue', '$\\bf{Greedy}$'],
        }
    plot_prior = {
            'linucb': 1,
            'lints': 2,
            'greedy': 3,
            'rb': 4,
        }
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    cat = os.listdir(root)
    paths = []
    for c in cat:
        folders = os.listdir(root+c)
        for folder in folders:
            paths.append(root + c + '/' + folder + '/')  
    for path in paths:
        if 'overc' in path: continue
        attack = path.split('/')[-2]
        fn = path.split('/')[-3]
        plot_name = fn + '_' + attack
        attack = attack_figure_name(attack)
        fn = fn.capitalize()
        title = '$\\bf{' + fn + ',\ ' + attack + '}$'
        print('plotting data in {}, output in {}.pdf'.format(path, plot_name))
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 12, 'axes.labelsize': 18, 'legend.fontsize': 18,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8), 'figure.max_open_warning': 0}
        pylab.rcParams.update(params)
        leg = []
        keys = os.listdir(path)
        keys = sorted(keys, key=lambda kv: plot_prior[kv])
        y_label = 'Cumulative Regret'
        limity = 0
        for key in keys:
            if key not in plot_style.keys(): continue
            leg += [plot_style[key][-1]]
            avg = np.loadtxt(path+key)
            if key != 'greedy': limity = max(limity, avg[-1])
            plot.plot((list(range(T))), avg, 'k', linestyle = plot_style[key][0], color = plot_style[key][1], linewidth = 2)
            plot.legend((leg), loc='upper left', fontsize=16, frameon=False)
        plot.xlabel('Iterations')
        plot.ylabel(y_label)
        limity = (int(limity / 10**4) + 1) * 10**4
        plot.ylim([0, limity])
        plot.title(title)
        fig.savefig('plots/' + plot_name + '.pdf', dpi=300, bbox_inches = "tight")
        
draw_C()
draw_figure()