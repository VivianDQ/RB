import numpy as np
import random
import math
import os
import argparse
import importlib
from algorithms.LinUCB import *
from algorithms.LinUCB_oracle import *
from algorithms.LinTS import *
from algorithms.Greedy import *
from algorithms.robustBandit import *

parser = argparse.ArgumentParser(description='experiments for RobustBandit')
parser.add_argument('-type', '--type', type=str, default = 'reward', help = 'attack reward or context')
parser.add_argument('-attack', '--attack', type=str, default = 'flip_theta', help = 'type of attacks')
parser.add_argument('-budget', '--budget', type=float, default = 100, help = 'attack budget')
parser.add_argument('-data', '--data', type=str, default = 'simulations', help = 'can be netflix or movielens')
args = parser.parse_args()

T = 10**6
d = 10
K = 20
C = 100
rep = 10
lamda = 0.1
delta = 0.01
typ = args.type
attack = args.attack
datatype = args.data

LinUCB = eval('LinUCB_{}'.format(typ))
LinUCB_oracle = eval('LinUCB_oracle_{}'.format(typ))
Greedy = eval('Greedy_{}'.format(typ))
LinTS = eval('LinTS_{}'.format(typ))
robustBandit = eval('robustbandit_{}'.format(typ))
data_generator = importlib.import_module('algorithms.data_generator_attack_{}'.format(typ))
context = data_generator.context
movie = data_generator.movie

# create the path for saving numerical results based on algo name and attack type
if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + attack + '/'):
    os.mkdir('results/' + datatype + '/' + attack + '/')
path = 'results/' + datatype + '/' + attack + '/'

reg_linucb = np.zeros(T)
reg_lints = np.zeros(T)
reg_greedy = np.zeros(T)
reg_rb = np.zeros(T)

if datatype == 'movielens' or datatype == 'netflix':
    # check whether real data files exist:
    if not os.path.isfile('data/{}_users_matrix_d{}'.format(datatype, d)) or not os.path.isfile('data/{}_movies_matrix_d{}'.format(datatype, d)):
        print("{holder} data does not exist, will run preprocessing for {holder} data now. Note that you need to put the raw data specified in readme file in order to get the preprocessing done. If you are running experiments for netflix data, then preprocessing takes a long time".format(holder=datatype))
        from data.preprocess_data import *
        process = eval("process_{}_data".format(datatype))
        process()
        print("real data processing done")   
        
    users = np.loadtxt("data/{}_users_matrix_d{}".format(datatype, d))
    fv = np.loadtxt("data/{}_movies_matrix_d{}".format(datatype, d))
    np.random.seed(0)
    thetas = np.zeros((rep, d))
    print(users.shape, fv.shape)
    for i in range(rep):
        # true model para is averaged over randomly selected 100 users features
        thetas[i,:] = np.mean(users[np.random.choice(len(users), 100, replace = False), :], axis=0)

for i in range(rep):
    print(i, ": ", end = " ")
    np.random.seed(i+1)
    if datatype == 'simulations':
        ub = 1/math.sqrt(d)
        lb = -1/math.sqrt(d)
        theta = np.random.uniform(lb, ub, d)
        fv = np.random.uniform(lb, ub, (T, K, d))
        bandit = context(C, K, lb, ub, T, d, true_theta = theta, fv=fv)
    elif datatype in ['movielens', 'netflix']:
        theta = thetas[i, :]
        bandit = movie(C, T, theta, K = K, d = d, fv = fv)   
    bandit.build_bandit()
    
    linucb = LinUCB(attack, bandit, T)
    reg_linucb += linucb.linucb(delta, lamda)
    
    rb = robustBandit(attack, bandit, T)
    reg_rb += rb.robustbandit(delta, lamda)
    
    ts = LinTS(attack, bandit, T)
    reg_lints += ts.lints(delta, lamda)
    
    grdy = Greedy(attack, bandit, T)
    reg_greedy += grdy.greedy(lamda)

    print("linucb {}, rb {}, ts {}, greedy {}".format(reg_linucb[-1], reg_rb[-1], reg_lints[-1], reg_greedy[-1]))
    result = {
        'linucb': reg_linucb/(i+1),
        'rb': reg_rb/(i+1),
        'lints': reg_lints/(i+1),
        'greedy': reg_greedy/(i+1),
    }
    for k,v in result.items():
        np.savetxt(path + k, v)     