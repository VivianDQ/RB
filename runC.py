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

parser = argparse.ArgumentParser(description='regrets over budgets')
parser.add_argument('-type', '--type', type=str, default = 'reward', help = 'attack reward or context')
parser.add_argument('-attack', '--attack', type=str, default = 'flip_theta', help = 'type of attacks')
args = parser.parse_args()

T = 10**6
d = 10
K = 20
rep = 10
lamda = 0.1
delta = 0.01
Cs = [0, 100, 10**3, 10**4, 10**5]
datatype = 'simulations'
attack = args.attack
typ = args.type

LinUCB = eval('LinUCB_{}'.format(typ))
LinUCB_oracle = eval('LinUCB_oracle_{}'.format(typ))
Greedy = eval('Greedy_{}'.format(typ))
LinTS = eval('LinTS_{}'.format(typ))
robustBandit = eval('robustbandit_{}'.format(typ))
data_generator = importlib.import_module('algorithms.data_generator_attack_{}'.format(typ))
context = data_generator.context
movie = data_generator.movie

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + 'overc' + '/'):
    os.mkdir('results/' + 'overc' + '/')
path = 'results/overc/'

result = {
    'linucb': np.zeros(len(Cs)),
    'rb': np.zeros(len(Cs)),
    'lints': np.zeros(len(Cs)),
    'greedy': np.zeros(len(Cs)),
}
    
for cindex, C in enumerate(Cs):
    for i in range(rep):
        np.random.seed(i+1)
        ub = 1/math.sqrt(d)
        lb = -1/math.sqrt(d)
        theta = np.random.uniform(lb, ub, d)
        fv = np.random.uniform(lb, ub, (T, K, d))
        bandit = context(C, K, lb, ub, T, d, true_theta = theta, fv=fv)
        bandit.build_bandit()

        linucb = LinUCB(attack, bandit, T)
        result['linucb'][cindex] += linucb.linucb(delta, lamda)[-1]

        rb = robustBandit(attack, bandit, T)
        result['rb'][cindex] += rb.robustbandit(delta, lamda)[-1]

        ts = LinTS(attack, bandit, T)
        result['lints'][cindex] += ts.lints(delta, lamda)[-1]

        grdy = Greedy(attack, bandit, T)
        result['greedy'][cindex] += grdy.greedy(lamda)[-1]
    print('experiments for budget = {} done'.format(C))
        
for k,v in result.items():
    result[k] /= rep
file = open(path + attack + '.txt',"wb")
pickle.dump(result, file)
file.close()
print(result)