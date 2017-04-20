from __future__ import division, print_function
import warnings
import os

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set_style()

from scipy.optimize import minimize
from scipy.misc import factorial
from multiprocessing import pool

def process_raw_data(raw,filestr='data/PL_2015-16.csv'):
    """Data downloaded from http://www.football-data.co.uk/englandm.php

    Arguments
    ---------

    raw: pandas.DataFrame
        The data as downloaded from the above url, loaded into pandas.

    filestr: (='data/PL_2015-16.csv') string
        Location to save the modified file at.

    Returns
    -------

    data: pandas.DataFrame
        The data indexed by date and with columns for home and away teams, home and 
        away goals."""

    raw.loc[:,'Date'] = pd.to_datetime(raw.loc[:,'Date'],dayfirst=True)
    raw.rename(columns={'FTHG':'HomeGoals','FTAG':'AwayGoals','B365H':'BookiesHomeWin',\
                        'B365D':'BookiesDraw', 'B365A':'BookiesAwayWin'},inplace=True)
    raw = raw[['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals','BookiesHomeWin','BookiesDraw','BookiesAwayWin']]
    raw.loc[:,'BookiesHomeWin'] = 1./raw['BookiesHomeWin']
    raw.loc[:,'BookiesDraw'] = 1./raw['BookiesDraw']
    raw.loc[:,'BookiesAwayWin'] = 1./raw['BookiesAwayWin']
    raw = raw.set_index('Date')
    if filestr is not None: raw.to_csv(filestr)

    return raw

def load_and_preprocess():

    """
    Load the data and do some pre-processing. The data are 
    presumed to be located at "data/PL_2015-16.csv".

    Returns
    -------

    data: pandas.DataFrame
        The data, with an extra 'Gameweek' column added 
        for convenience, and teams replaced by integer 
        identifiers.

    teams: array_like
        List of teams such that the integer identifier for 
        teams[i] is i. 


    """

    data = pd.read_csv("data/PL_2015-16.csv")
    teams = pd.unique(data['HomeTeam'])
    for i,team in enumerate(teams):
        home_idx = data['HomeTeam'] == team 
        away_idx = data['AwayTeam'] == team
        data.loc[home_idx,'HomeTeam'] = i
        data.loc[away_idx,'AwayTeam'] = i
    data[['HomeTeam','AwayTeam']] = data[['HomeTeam','AwayTeam']].apply(pd.to_numeric)

    data = data.set_index('Date')
    data.index = pd.to_datetime(data.index)
    data['Gameweek'] = data.index.year * 100 + data.index.week
    data['Gameweek'] = data['Gameweek'].diff(1).fillna(1.)
    data.loc[data['Gameweek'] != 0, 'Gameweek'] = 1.
    data['Gameweek'] = data['Gameweek'].cumsum().astype(int)

    return data,teams

def compute_lambdas_mus(params,data,n_teams):

    """
    Compute the rates for the home team and away 
    team scoring.

    Arguments
    ---------

    params: array_like
        The model parameters.

    data: pandas.DataFrame
        The data.

    n_teams: int
        The number of teams in the league (20 for the PL).

    Returns
    -------

    mu_k: array_like
        The rates for the away teams. 

    lambda_k: array_like
        The rates for the home teams.

    """

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

def log_likelihood(params,data,n_teams=20,zeta=0.003,date=None):

    """
    Compute the (negative) log likelihood for a given set of model 
    parameters.

    Arguments
    ---------

    params: array_like
        The model parameters.

    data: pandas.DataFrame
        The data.

    n_teams: (=20) int
        The number of teams in the league (20 for the PL).

    zeta: float 
        The hyperparameter that determines the importance of 
        matches at time t in the past. The rate is in 1/days 
        rather than 1/half-weeks as is done in Dixon and Coles.

    date: string or datetime object 
        The date for which the fit should be carried out.

    Returns
    -------

    lnL: float
        The negative log-likelihood. 

    """

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
        prediction_date = np.datetime64(pd.to_datetime(date))
    dt = (prediction_date - match_dates).astype('timedelta64[D]')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        lnL = -np.sum( (-mu_k + y_k*np.log(mu_k) - lambda_k + x_k*np.log(lambda_k) + np.log(correlations))*np.exp(-dt.astype(float)*zeta) )

    return lnL

def rho_constraint_low(params,data,n_teams):

    """
    The lower bound on the correlation parameter from 
    Dixon and Coles.

    Arguments
    ---------

    params: array_like
        The model parameters.

    data: pandas.DataFrame
        The data.

    n_teams: int 
        The number of teams in the league.

    Returns
    -------

    bound: float
        rho - lower_bound, such that bound > 0 must be satisfied.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        mu_k,lambda_k = compute_lambdas_mus(params,data,n_teams)
        lower_bound = np.max(np.hstack((-1./lambda_k,-1./mu_k)))
    return params[-1] - lower_bound

def rho_constraint_high(params,data,n_teams):

    """
    The upper bound on the correlation parameter from 
    Dixon and Coles.

    Arguments
    ---------

    params: array_like
        The model parameters.

    data: pandas.DataFrame
        The data.

    n_teams: int 
        The number of teams in the league.

    Returns
    -------

    bound: float
        upper_bound - rho, such that bound > 0 must be satisfied.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        mu_k,lambda_k = compute_lambdas_mus(params,data,n_teams)
        upper_bound = np.min(np.append((lambda_k*mu_k)**-1.,1.))
    return upper_bound - params[-1]


def compute_parameters(data,params_guess=None,n_teams=20,date=None,zeta=0.003):

    """
    Compute the parameters of the model given some data.

    Arguments
    ---------

    data: pandas.DataFrame
        The data.

    params_guess: (= None) array_like
        Guesses for the model parameters. If None, then all the parameters 
        are set to 1 except for rho, which is set to 0.

    n_teams: (=20) int
        The number of teams in the league (20 for the PL).

    date: (=None) string or datetime object 
        The date for which the fit should be carried out.

    zeta: (=0.003) float 
        The hyperparameter that determines the importance of 
        matches at time t in the past. The rate is in 1/days 
        rather than 1/half-weeks as is done in Dixon and Coles.

    Returns
    -------

    alphas: array_like
        The attacking indices for each team.

    betas: array_like
        The defending indices for each team.

    gamma: float
        The home advantage parameter.

    rho: float
        The correlation parameter for low-scoring 
        games. 

    """

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
        Exception('Failed to optimize the model.')

    alphas = res.x[:n_teams - 1]
    alphas = np.append(alphas, n_teams - np.sum(alphas))

    betas = res.x[n_teams - 1 : 2*n_teams - 1]
    gamma = res.x[-2]
    rho = res.x[-1]

    return alphas,betas,gamma,rho


def fit_model_up_to_round(data,gameweek=1,window=10):

    """
    Fit the model up to a given 'round', defined by the Gameweek column 
    in the data file. Instead of using exp(-zeta * t), a simple sliding 
    window function is used where the twenty previous gameweeks are considered.

    Arguments
    ---------

    data: pandas.DataFrame
        The data.

    gameweek: (=1) int
        The gameweek (1<= gameweek <= 36) up to which the model should be fit. Only the ten 
        weeks before the selected gameweek will be considered.

    window: (=10) int
        The number of gameweeks to consider when fitting the model up to the given gameweek.

    Returns
    -------

    alphas: array_like
        The attacking indices for each team.

    betas: array_like
        The defending indices for each team.

    gamma: float
        The home advantage parameter.

    rho: float
        The correlation parameter for low-scoring 
        games. 

    """

    params = np.ones(41)
    params[-1] = 0.
    if gameweek<window:
        train = data.loc[data['Gameweek'] <= gameweek, :]
    else:
        train = data.loc[(data['Gameweek'] <= gameweek)&(data['Gameweek'] > gameweek - window), :]
    res = compute_parameters(train,params_guess=params,n_teams=20,zeta=0.)
    alphas,betas,gamma,rho = res
    return alphas,betas,gamma,rho

def fit_model_season(data,window=10):

    """
    Fit the model for the whole season, computing the parameters for each gameweek. Note 
    that the first ~ 5 gameweeks should really be disregarded because the parameter estimates 
    are fluctuating rapidly because of a lack of historical data.

    Arguments
    ---------

    data: pandas.DataFrame
        The data.

    window: (=10) int
        The number of previous gameweeks to consider when fitting the model up 
        to a given gameweek.

    Returns
    -------

    alphas: array_like
        The attacking indices for each team. The shape of this 
        array will be [n_gameweeks, n_teams] such that alphas[i,j] 
        gives the attacking index of team j in gameweek i.

    betas: array_like
        The defending indices for each team. The shape of this 
        array will be [n_gameweeks, n_teams] such that betas[i,j] 
        gives the defending index of team j in gameweek i.

    gamma: float
        The home advantage parameter. The shape of this array will 
        be [n_gameweeks,], such that gamma[i] is the home advantage 
        parameter in gameweek i.

    rho: float
        The correlation parameter for low-scoring games. The shape 
        of this array will be [n_gameweeks,], such that rho[i] 
        is the correlation parameter in gameweek i.
    """

    alpha_tot = np.zeros((36,20))
    beta_tot = np.zeros((36,20))
    gamma_tot = np.zeros(36)
    rho_tot = np.zeros(36)

    for i in np.arange(1,37):
        if i==1:
            alpha_tot[0,:],beta_tot[0,:],gamma_tot[0],rho_tot[0] = fit_model_up_to_round(data,window=window)
        else:
            try:
                alpha_tot[i-1,:],beta_tot[i-1,:],gamma_tot[i-1],rho_tot[i-1] = fit_model_up_to_round(data,gameweek=i,window=window)
            except:
                alpha_tot[i-1,:],beta_tot[i-1,:],gamma_tot[i-1],rho_tot[i-1] = alpha_tot[i-2,:],beta_tot[i-2,:],gamma_tot[i-2],rho_tot[i-2]

    return alpha_tot,beta_tot,gamma_tot,rho_tot

def model(x,y,alpha_i,alpha_j,beta_i,beta_j,gamma,rho):

    """
    The basic probability model of Dixon and Coles. Computes the 
    probability of the home team scoring x goals and the away team 
    scoring y goals given the appropriate model parameters. Note 
    that if the inputs are arrays, they must all have the same shape.

    Arguments
    ---------

    x: float or array_like
        Home team goals.

    y: float or array_like
        Away team goals. 

    alpha_i: float or array_like
        Home team attacking indices.

    alpha_j: float or array_like
        Away team attacking indices.

    beta_i: float or array_like
        Home team defending indices.

    beta_j: float or array_like
        Away team defending indices.

    gamma: float or array_like
        Home team advantage parameters.

    rho: float or array_like
        Correlation parameters.

    Returns
    -------

    probability: float or array_like
        The probability of the home team scoring x goals and the 
        away team scoring y goals.

    """

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

    """
    The probabilities of home wins, draws and away wins given a set of 
    model parameters. Computed by constructing a grid of home and away 
    goals between 0 and 10 and summing over the appropriate elements. 
    Note that this is not vectorized, so that one must loop over different 
    sets of parameters in order to obtain probabilities for different 
    matches.

    Arguments
    ---------

    alpha_i: float
        Home team attacking indices.

    alpha_j: float
        Away team attacking indices.

    beta_i: float
        Home team defending indices.

    beta_j: float
        Away team defending indices.

    gamma: float
        Home team advantage parameters.

    rho: float
        Correlation parameters.    

    Returns
    -------

    home_win: float
        The predicted probability of a home win.

    draw: float
        The predicted probability of a draw.

    away_win: float
        The predicted probability of an away win.

    """
 
    n_goals = np.arange(0,11)
    x,y = np.meshgrid(n_goals,n_goals,indexing='ij')
    probs = model(x,y,alpha_i,alpha_j,beta_i,beta_j,gamma,rho)
    home_win = np.sum(probs[x>y])
    away_win = np.sum(probs[x<y])
    draw = np.sum(probs[x==y])

    return home_win,draw,away_win

def result_likelihood(results,home_prob,draw_prob,away_prob):

    """
    The predictive likelihood function (Equation 4.7 of Dixon and Coles) 
    used to constrain the hyperparameter zeta.

    Arguments
    ---------

    results: array_like
        An array with elements with values 0, 1 or 2.
        0 => home win
        1 => draw 
        2 => away win 

    home_prob: array_like
        The predicted probability of a home win. 

    draw_prob: array_like
        The predicted probability of a draw. 

    away_prob: array_like
        The predicted probability of an away win. 

    Returns
    -------

    lnprob: float
        The predictive log likelihood.

    """

    lnprob = np.ones(results.shape)
    lnprob[results==0] = np.log(home_prob[results==0])
    lnprob[results==1] = np.log(draw_prob[results==1])
    lnprob[results==2] = np.log(away_prob[results==2])

    return np.sum(lnprob)

def compute_zeta(data,zeta = np.linspace(0.,0.015,20) ):

    """
    Given some data, calculate the predictive log-likelihood for a 
    grid of zeta values in order to find the optimal value. The data 
    are divided into unique 'game days'. For each game day, the model 
    is trained on all preceding game days, and the result probabilities 
    stored. For each zeta, the predictive log-likelihood is then calculated.

    Arguments
    ---------

    data: pandas.DataFrame
        The data. 

    zeta = (= numpy.linspace(0.,0.015,20) ) array_like
        Range of zeta values to consider. 

    Returns
    -------

    zeta: array_like
        The zeta values considered.

    lnprob: array_like
        The predictive log-likelihood for each zeta.

    """

    results = np.ones(len(data)).astype(int)
    x = data['HomeGoals'].values
    y = data['AwayGoals'].values 
    results[x>y] = 0
    results[x==y] = 1 
    results[x<y] = 2

    unique_dates = np.unique(data.index.values)[20:]
    lnprob = np.zeros_like(zeta)

    for i,zi in enumerate(zeta):
        if i!=0:
            print('\033[F' + '\033[K' + '\033[F')
        print("{} zeta values to go...".format(len(zeta) - i))
        home_probs = np.zeros(results.shape)
        draw_probs = np.zeros(results.shape)
        away_probs = np.zeros(results.shape)
        for date in unique_dates:
            thisdate = data.index==date
            training_thisdate = data.loc[data.index<date,:]
            try:
                params_guess = np.append(np.append(np.hstack((a[:19],b)),g),r)
            except:
                params_guess = np.ones(41)
                params_guess[-1] = 0.
            a,b,g,r = compute_parameters(training_thisdate,params_guess=params_guess,n_teams=20,date=date,zeta=zi)
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
        ind = home_probs!=0.
        home_probs,draw_probs,away_probs,res_thisdate = home_probs[ind],draw_probs[ind],away_probs[ind],results[ind]
        lnprob[i] = result_likelihood(res_thisdate,home_probs,draw_probs,away_probs)
    
    return zeta,lnprob

def calculate_odds(data,zeta=0.003):

    """
    Given some data and an optimal value of zeta, compute and store the 
    predicted probabilities of home wins, draws and away wins for each 
    match. The first twenty game days are only used to fit the model: 
    predictions are not made for these because the model is too uncertain 
    with such a small quantity of data.

    Arguments
    ---------

    data: pandas.DataFrame
        The data.

    zeta: (=0.003) float
        The zeta value to use when fitting the model on a given day.

    Returns
    -------

    data: pandas.DataFrame
        The data, with four new columns (ProbHomeWin,ProbDraw,ProbAwayWin,Result).
        The result column takes values 0 (home win), 1 (draw), 2 (away win).

    """

    unique_dates = np.unique(data.index.values)[20:]

    results = np.ones(len(data)).astype(int)
    x = data['HomeGoals'].values
    y = data['AwayGoals'].values 
    results[x>y] = 0
    results[x==y] = 1 
    results[x<y] = 2

    home_probs = np.zeros(len(data))
    draw_probs = np.zeros(len(data))
    away_probs = np.zeros(len(data))
    for date in unique_dates:
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

    ind = home_probs==0.
    home_probs[ind] = np.nan
    draw_probs[ind] = np.nan
    away_probs[ind] = np.nan
    
    data.loc[:,'ProbHomeWin'] = pd.Series(home_probs,index=data.index)
    data.loc[:,'ProbDraw'] = pd.Series(draw_probs,index=data.index)
    data.loc[:,'ProbAwayWin'] = pd.Series(away_probs,index=data.index)
    data.loc[:,'Result'] = pd.Series(results,index=data.index)

    return data



