import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import pystan 

from adjustText import adjust_text

def load_data():
    #Load the data for the PL season 2016/17
    data = pd.read_csv('data/PL_2016_17.csv')
    prior_data = pd.read_csv('data/mcnulty_predictions.csv')
    teams = np.unique(data['HomeTeam'].values)
    for i,team in enumerate(teams):
        home_idx = data['HomeTeam'] == team 
        away_idx = data['AwayTeam'] == team
        data.loc[home_idx,'HomeTeam'] = i+1
        data.loc[away_idx,'AwayTeam'] = i+1
    data[['HomeTeam','AwayTeam']] = data[['HomeTeam','AwayTeam']].apply(pd.to_numeric)
    return data, prior_data, teams

def fit_model():
    #fit the model using pystan
    data, prior_data, teams = load_data()
    pl_data = {'nteams' : len(teams),
               'ngames' : len(data),
               'home_goals' : data['FTHG'].values,
               'away_goals' : data['FTAG'].values,
               'home_st': data['HST'].values,
               'away_st': data['AST'].values,
               'home_team' : data['HomeTeam'].values,
               'away_team' : data['AwayTeam'].values,
               'mcnulty_rank': np.argsort(prior_data['Team'].values)+1
               }
    sm = pystan.StanModel(file='premierleague.stan')
    fit = sm.sampling(data=pl_data, iter=5000, chains=4)
    return fit


def plot_alpha_beta(a, b, teams):
    #plot the alpha and beta distribution of the teams
    fig, ax = plt.subplots()
    a_mean, b_mean = np.mean(a, axis=0), np.mean(b, axis=0)
    ax.scatter(a_mean, b_mean)
    texts = []
    for ai,bi,s in zip(a_mean, b_mean, teams):
        texts.append(plt.text(ai, bi, s, size=7))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.))
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.errorbar(1.25, 1.3, xerr=np.mean(np.std(a,axis=0)), yerr=np.mean(np.std(b,axis=0)), fmt='o', capthick=0.01)
    plt.gca().arrow(1.44, 1.64, 0., -0.2, head_width=0.02, head_length=0.02, fc='k', ec='k')
    plt.text(1.38, 1.65, "Better \ndefense", size=7)
    plt.gca().arrow(0.6, 0.8, 0.2, 0., head_width=0.02, head_length=0.02, fc='k', ec='k')
    plt.text(0.6, 0.82, "Better attack", size=7)
    plt.text(1.27, 1.23, "Typical \nuncertainty", size=7)
    return None






