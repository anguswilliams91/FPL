from __future__ import division, print_function

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set_style()

from scipy.optimize import minimize

def load_and_preprocess():

    data = pd.read_csv("data/PL_2015-16.csv")
    teams = pd.unique(data['HomeTeam'])
    for i,team in enumerate(teams):
        home_idx = data['HomeTeam'] == team 
        away_idx = data['AwayTeam'] == team
        data.loc[home_idx,'HomeTeam'] = i
        data.loc[away_idx,'AwayTeam'] = i
    data[['HomeTeam','AwayTeam']] = data[['HomeTeam','AwayTeam']].apply(pd.to_numeric)
    return data,teams

def log_likelihood(params,data,n_teams=20):

    alphas = params[: n_teams-1 ]
    betas = params[n_teams-1 : 2*n_teams - 1]
    gamma = params[-1]
    alphas = np.append(alphas,n_teams-np.sum(alphas))

    alpha_ik = alphas[data['HomeTeam'].values]
    beta_ik  = betas[data['HomeTeam'].values]
    alpha_jk = alphas[data['AwayTeam'].values]
    beta_jk = betas[data['AwayTeam'].values]


    x_k = data['HomeGoals'].values 
    y_k = data['AwayGoals'].values

    mu_k = alpha_jk * beta_ik
    lambda_k = alpha_ik * beta_jk * gamma

    return -np.sum( -mu_k + y_k*np.log(mu_k) - lambda_k + x_k*np.log(lambda_k) )

def jacobian(params,data,n_teams=20):

    alphas = params[: n_teams-1 ]
    betas = params[n_teams-1 : 2*n_teams - 1]
    gamma = params[-1]
    alphas = np.append(alphas,n_teams-np.sum(alphas))

    alpha_ik = alphas[data['HomeTeam'].values]
    beta_ik  = betas[data['HomeTeam'].values]
    alpha_jk = alphas[data['AwayTeam'].values]
    beta_jk = betas[data['AwayTeam'].values]

    x_k = data['HomeGoals'].values 
    y_k = data['AwayGoals'].values

    mu_k = alpha_jk * beta_ik
    lambda_k = alpha_ik * beta_jk * gamma

    deriv_alphas = np.zeros(n_teams-1)
    deriv_betas = np.zeros(n_teams)
    for i in np.arange(n_teams - 1):
        deriv_alphas[i] = np.sum( (gamma*beta_jk*(x_k/lambda_k - 1.))[data['HomeTeam'].values == i] ) +\
                            np.sum( (beta_ik*(y_k/mu_k - 1.))[data['AwayTeam'].values == i] )
        deriv_betas[i] = np.sum( (gamma*alpha_ik*(x_k/lambda_k - 1.))[data['AwayTeam'].values == i] ) +\
                            np.sum( (alpha_jk*(y_k/mu_k - 1.))[data['HomeTeam'].values == i] )

    deriv_betas[-1] = np.sum( (gamma*alpha_ik*(x_k/lambda_k - 1.))[data['AwayTeam'].values == n_teams - 1] ) +\
                            np.sum( (alpha_jk*(y_k/mu_k - 1.))[data['HomeTeam'].values == n_teams - 1] )
    deriv_gamma = np.sum( (x_k/lambda_k - 1.)*alpha_ik*beta_jk )
    derivs = np.hstack((deriv_alphas,deriv_betas))
    derivs = np.append(derivs,deriv_gamma)

    return -derivs

def compute_parameters(params_guess,data,n_teams=20):

    cons = ({'type': 'ineq', 'fun': lambda x:  n_teams - np.sum(x[: n_teams - 1]) })

    bnds = ((0.,None),)*2*n_teams

    res = minimize(log_likelihood, params_guess, args=data, jac=jacobian, \
            method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 200})

    if res.success != True:
        print('Failed to optimize the model.')
        return res

    alphas = res.x[:n_teams - 1]
    alphas = np.append(alphas, n_teams - np.sum(alphas))

    betas = res.x[n_teams - 1 : 2*n_teams - 1]
    gamma = res.x[-1]

    return alphas,betas,gamma
