from __future__ import division, print_function

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set_style()

from scipy.optimize import minimize
from scipy.misc import factorial
from multiprocessing import pool

def process_raw_data(raw,filestr='data/PL_2015-16.csv'):
    """http://www.football-data.co.uk/englandm.php"""

    raw.loc[:,'Date'] = pd.to_datetime(raw.loc[:,'Date'],dayfirst=True)
    raw.rename(columns={'FTHG':'HomeGoals','FTAG':'AwayGoals'},inplace=True)
    raw = raw[['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals']]
    raw = raw.set_index('Date')
    if filestr is not None: raw.to_csv(filestr)

    return raw

def load_and_preprocess():

    data = pd.read_csv("data/PL_2015-16.csv")
    teams = pd.unique(data['HomeTeam'])
    for i,team in enumerate(teams):
        home_idx = data['HomeTeam'] == team 
        away_idx = data['AwayTeam'] == team
        data.loc[home_idx,'HomeTeam'] = i
        data.loc[away_idx,'AwayTeam'] = i
    data[['HomeTeam','AwayTeam']] = data[['HomeTeam','AwayTeam']].apply(pd.to_numeric)

    #add a 'round' column by grouping by week
    data = data.set_index('Date')
    data.index = pd.to_datetime(data.index)
    data['Gameweek'] = data.index.year * 100 + data.index.week
    data['Gameweek'] = data['Gameweek'].diff(1).fillna(1.)
    data.loc[data['Gameweek'] != 0, 'Gameweek'] = 1.
    data['Gameweek'] = data['Gameweek'].cumsum().astype(int)

    return data,teams

def compute_lambdas_mus(params,data,n_teams):

    alphas = params[: n_teams-1 ]
    betas = params[n_teams-1 : 2*n_teams - 1]
    gamma = params[-2]
    rho = params[-1]
    alphas = np.append(alphas,n_teams-np.sum(alphas))

    alpha_ik = alphas[data['HomeTeam'].values]
    beta_ik  = betas[data['HomeTeam'].values]
    alpha_jk = alphas[data['AwayTeam'].values]
    beta_jk = betas[data['AwayTeam'].values]

    x_k = data['HomeGoals'].values.astype('float') 
    y_k = data['AwayGoals'].values.astype('float')

    mu_k = alpha_jk * beta_ik
    lambda_k = alpha_ik * beta_jk * gamma

    return mu_k,lambda_k

def log_likelihood(params,data,n_teams=20,zeta=0.0,date=None):

    rho = params[-1]
    x_k = data['HomeGoals'].values.astype('float') 
    y_k = data['AwayGoals'].values.astype('float')
    mu_k,lambda_k = compute_lambdas_mus(params,data,n_teams)

    correlations = np.ones_like(x_k)
    nil_nil = (x_k==0)&(y_k==0)
    one_nil = (x_k==1)&(y_k==0)
    nil_one = (x_k==0)&(y_k==1)
    one_one = (x_k==1)&(y_k==1)
    correlations[nil_nil] = (1. - lambda_k*mu_k*rho)[nil_nil]
    correlations[one_nil] = (1. + mu_k*rho)[one_nil]
    correlations[nil_one] = (1. + lambda_k*rho)[nil_one]
    correlations[one_one] = (1. - rho)

    match_dates = data.index.values
    if date is None:
        prediction_date = data.index.values[-1]
    else:
        prediction_date = pd.to_datetime(date)
    dt = np.array([pd.Timedelta(prediction_date - mdi).days/7. for mdi in match_dates])

    return -np.sum( (-mu_k + y_k*np.log(mu_k) - lambda_k + x_k*np.log(lambda_k) + np.log(correlations))*np.exp(-dt*zeta))

def rho_constraint_low(params,data,n_teams):

    mu_k,lambda_k = compute_lambdas_mus(params,data,n_teams)
    lower_bound = np.max(np.hstack((-1./lambda_k,-1./mu_k)))
    return params[-1] - lower_bound

def rho_constraint_high(params,data,n_teams):

    mu_k,lambda_k = compute_lambdas_mus(params,data,n_teams)
    upper_bound = np.min(np.append((lambda_k*mu_k)**-1.,1.))
    return upper_bound - params[-1]


def compute_parameters(data,params_guess=None,n_teams=20,date=None,zeta=0.):

    if params_guess is None:
        params_guess = np.ones(2*n_teams + 1)
        params_guess[-1] = 0.

    cons = ({'type': 'ineq', 'fun': lambda x:  n_teams - np.sum(x[: n_teams - 1]) },\
            {'type': 'ineq', 'fun': rho_constraint_low, 'args': (data,n_teams) },\
            {'type': 'ineq', 'fun': rho_constraint_high, 'args': (data,n_teams) } )

    bnds = ((0.,None),)*2*n_teams
    bnds = bnds + ((-1.,1.),)

    res = minimize(log_likelihood, params_guess, args=(data,n_teams,zeta,date), jac=None, \
            method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 200})

    if res.success != True:
        print('Failed to optimize the model.')
        return res

    alphas = res.x[:n_teams - 1]
    alphas = np.append(alphas, n_teams - np.sum(alphas))

    betas = res.x[n_teams - 1 : 2*n_teams - 1]
    gamma = res.x[-2]
    rho = res.x[-1]

    return alphas,betas,gamma,rho


def fit_model_up_to_round(data,gameweek=1):

    params = np.ones(41)
    params[-1] = 0.
    if gameweek<10:
        train = data.loc[data['Gameweek'] <= gameweek, :]
    else:
        train = data.loc[(data['Gameweek'] <= gameweek)&(data['Gameweek'] > gameweek - 10), :]
    res = compute_parameters(train,params_guess=params,n_teams=20)
    try:
        alphas,betas,gamma,rho = res
        return alphas,betas,gamma,rho
    except:
        return res

def fit_model_season(data):

    alpha_tot = np.zeros((62,20))
    beta_tot = np.zeros((62,20))
    gamma_tot = np.zeros(62)
    rho_tot = np.zeros(62)

    for i in np.arange(1,63):
        if i==1:
            alpha_tot[0,:],beta_tot[0,:],gamma_tot[0],rho_tot[0] = fit_model_up_to_round(data)
        else:
            try:
                alpha_tot[i-1,:],beta_tot[i-1,:],gamma_tot[i-1],rho_tot[i-1] = fit_model_up_to_round(data,gameweek=i)
            except:
                alpha_tot[i-1,:],beta_tot[i-1,:],gamma_tot[i-1],rho_tot[i-1] = alpha_tot[i-2,:],beta_tot[i-2,:],gamma_tot[i-2],rho_tot[i-2]

    return alpha_tot,beta_tot,gamma_tot,rho_tot

def model(x,y,alpha_i,alpha_j,beta_i,beta_j,gamma,rho):

    lambda_k = alpha_i*beta_j*gamma
    mu_k = alpha_j*beta_i

    if hasattr(x,'__len__'):

        correlations = np.ones(x.shape)
        nil_nil = (x==0)&(y==0)
        one_nil = (x==1)&(y==0)
        nil_one = (x==0)&(y==1)
        one_one = (x==1)&(y==1)
        correlations[nil_nil] = (1. - lambda_k*mu_k*rho)
        correlations[one_nil] = (1. + mu_k*rho)
        correlations[nil_one] = (1. + lambda_k*rho)
        correlations[one_one] = (1. - rho)

    else:

        if x==0 and y==0:
            correlations = 1. - lambda_k*mu_k*rho 
        elif x==1 and y==0:
            correlations = 1. + mu_k*rho 
        elif x==0 and y==1:
            correlations = 1. + lambda_k*rho 
        elif x==1 and y==1:
            correlations = 1. - rho 
        else:
            correlations = 1.        

    return correlations*np.exp(-lambda_k)*lambda_k**x*np.exp(-mu_k)*mu_k**y / (factorial(x)*factorial(y))

def result_probabilities(alpha_i,alpha_j,beta_i,beta_j,gamma,rho):
 
    n_goals = np.arange(0,11)
    x,y = np.meshgrid(n_goals,n_goals,indexing='ij')
    probs = model(x,y,alpha_i,alpha_j,beta_i,beta_j,gamma,rho)
    home_win = np.sum(probs[x>y])
    away_win = np.sum(probs[x<y])
    draw = np.sum(probs[x==y])

    return home_win,draw,away_win

def result_likelihood(results,home_prob,draw_prob,away_prob):

    #results = 0 (home), 1 (draw), 2 (away)

    lnprob = np.ones(results.shape)
    lnprob[results==0] = np.log(home_prob[results==0])
    lnprob[results==1] = np.log(draw_prob[results==1])
    lnprob[results==2] = np.log(away_prob[results==2])

    return np.sum(lnprob)

def zeta_likelihood(zeta,data):

    results = np.ones(len(data)).astype(int)
    x = data['HomeGoals'].values
    y = data['AwayGoals'].values 
    results[x>y] = 0
    results[x==y] = 1 
    results[x<y] = 2

    unique_dates = np.unique(data.index.values)[50::2]

    home_probs = np.zeros(results.shape)
    draw_probs = np.zeros(results.shape)
    away_probs = np.zeros(results.shape)
    for date in unique_dates:
        print(date)
        thisdate = data.index==date
        training_thisdate = data.loc[data.index<date,:]
        try:
            params_guess = np.append(np.append(np.hstack((a[:19],b)),g),r)
        except:
            params_guess = np.ones(41)
            params_guess[-1] = 0.
        a,b,g,r = compute_parameters(training_thisdate,params_guess=params_guess,n_teams=20,date=date,zeta=zeta)
        hometeams_thisdate = data['HomeTeam'].iloc[thisdate].values
        awayteams_thisdate = data['AwayTeam'].iloc[thisdate].values
        results_thisdate = results[thisdate]
        home_probs_thisdate = np.zeros(results_thisdate.shape)
        draw_probs_thisdate = np.zeros(results_thisdate.shape)
        away_probs_thisdate = np.zeros(results_thisdate.shape)
        a_home = np.array(a[hometeams_thisdate])
        a_away = np.array(a[awayteams_thisdate])
        b_home = np.array(b[hometeams_thisdate])
        b_away = np.array(b[awayteams_thisdate])
        for j in np.arange(len(hometeams_thisdate)):
            home_probs_thisdate[j],draw_probs_thisdate[j],away_probs_thisdate[j] = result_probabilities(a_home[j],\
                                                                                    a_away[j],b_home[j],b_away[j],g,r)
        home_probs[thisdate] = home_probs_thisdate
        draw_probs[thisdate] = draw_probs_thisdate
        away_probs[thisdate] = away_probs_thisdate
    
    return home_probs[home_probs!=0.],draw_probs[home_probs!=0.],away_probs[home_probs!=0.],results[home_probs!=0.]






