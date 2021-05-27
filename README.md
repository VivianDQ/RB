# Robust Stochastic Linear Contextual Bandits UnderAdversarial Attacks

This repository is the official implementation of [Robust Stochastic Linear Contextual Bandits UnderAdversarial Attacks].


## Requirements

To run the code, you will need 

```
Python3, NumPy, Matplotlib, R
```


## Commands

### Figure 1

To get the results in Figure 1 in the paper, run the following command:

```
python3 run.py -type {typename} -attack {attack_type} -data {dataname}
```

In the above command, replace ``{typename}`` with ``reward`` or ``context`` to run experiments for attack of reward or attack of context respectively. 

For attack of reward, the code supports the following ``attack``, and you can replace ``{attack_type}`` in the above command with one of the following:
- ``garcelon_strongest``: it pushes the reward to a random noise when the pulled arm is not the worst arm at each round.
- ``garcelon_topN``: it pushes the reward to a random noise when the pulled arm is among the best half arms at each round.
- ``oracle_strongest``: it pushes the reward to ``reward of worst arm - 0.01`` when the pulled arm is not the worst arm at each round. 
- ``oracle_topN``: it pushes the reward to ``reward of worst arm - 0.01`` when the pulled arm is among the best half arms at each round.

Note that for ``oracle`` attack, if the true reward is already below the attacked reward, then it does not apply the attack.

For attack of context, the code supports the following ``attack``, and you can replace ``{attack_type}`` in the above command with one of the following:
- ``context_strongest``: it attacks the contextual features when the potential pulled arm is not the worst arm at each round.
- ``context_topN``: it attacks the contextual features when the potential pulled arm is among the best half arms at each round.

Finally, replace ``{dataname}`` with one of ``simulations``, ``netflix``, ``movielens`` to run the experiments for these datasets. For ``netflix`` and ``movielens``, you need the real datasets and preprocess them (see next section for details).


### Figure 2 in Appendix

To get the results in Figure 2 in the appendix of the paper, run the following command:
```
python3 run.py -type context -attack context_topN
python3 run.py -type reward -attack garcelon_topN
python3 run.py -type reward -attack oracle_topN
```

## Real datasets

Get the following raw data from movielens official website and Kaggle's website and put them in the ``raw_data`` folder inside the ``data`` folder.

- Movieslens 100k dataset: you need ``u.data`` file in this dataset.
- Netflix dataset from Kaggle: you need ``combined_data_{i}.txt`` for ``i=1,2,3,4,5``.

Since we run matrix factorization on the raw data, we also need the matrix factorization package `libpmf-1.41`. Get the package from its [official website](https://www.cs.utexas.edu/~rofuyu/libpmf/) and unzip the package and save the whole `libpmf-1.41`` folder inside the ``data`` folder. You may need to read the instructions of that package to compile the program.

If it is your first time to run the experiments on ``netflix`` or ``movielens`` dataset, then our code will automatically preprocess the raw data, and it may take a while. Note that if the raw data does not exist or is not in the correct path, our code will report error.



## Plots

Numerical results will be saved in the ``results`` folder which is automatically created by our code. To produce the same plots as in our paper, run the following command, it will create a ``plots`` folder and all the figures will be saved there.

```
python3 plot.py
```
