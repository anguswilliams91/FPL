data {
    int nteams;
    int ngames;
    int home_goals[ngames];
    int away_goals[ngames];
    int home_s[ngames];
    int away_s[ngames];
    int home_st[ngames];
    int away_st[ngames];
    int home_team[ngames];
    int away_team[ngames];
    vector[nteams] mcnulty_rank;
}
transformed data {
    vector[nteams] normalised_rank;
    normalised_rank = 2*((1. - mcnulty_rank)/19. + 0.5);
}
parameters {
    vector<lower=0>[nteams] a;
    vector<lower=0>[nteams] b;
    vector<lower=0, upper=1>[nteams] f;
    vector<lower=0, upper=1>[nteams] g;
    real<lower=0> gamma;
    real corr_log_a;
    real corr_log_b;
    real<lower=0> sigma_log_a;
    real<lower=0> sigma_log_b;
    real<lower=0> mu_log_b;
    real<lower=0> alpha_f;
    real<lower=0> beta_f;
    real<lower=0> alpha_g;
    real<lower=0> beta_g;
}
model {
    a ~ lognormal(corr_log_a*normalised_rank, sigma_log_a);
    b ~ lognormal(mu_log_b + corr_log_b*log(a), sigma_log_b);
    f ~ beta(alpha_f, beta_f);
    g ~ beta(alpha_g, beta_g);
    gamma ~ lognormal(0, 1);
    corr_log_a ~ normal(0, 1);
    corr_log_b ~ normal(0, 1);
    sigma_log_a ~ normal(0, 1);
    sigma_log_b ~ normal(0, 1);
    mu_log_b ~ normal(0, 1);
    alpha_f ~ normal(0, 10);
    beta_f ~ normal(0, 10);
    alpha_g ~ normal(0, 10);
    beta_g ~ normal(0, 10);
    for (i in 1:ngames) {
        home_s[i] ~ poisson(a[home_team[i]]*b[away_team[i]]*gamma);
        away_s[i] ~ poisson(a[away_team[i]]*b[home_team[i]]);
        home_st[i] ~ binomial(home_s[i], f[home_team[i]]);
        away_st[i] ~ binomial(away_s[i], f[away_team[i]]);
        home_goals[i] ~ binomial(home_st[i], g[away_team[i]]);
        away_goals[i] ~ binomial(away_st[i], g[home_team[i]]);
        }
}
generated quantities {
    int<lower=0> home_goals_sim[ngames];
    int<lower=0> away_goals_sim[ngames];
    int<lower=0> home_st_sim[ngames];
    int<lower=0> away_st_sim[ngames];
    int<lower=0> home_s_sim[ngames];
    int<lower=0> away_s_sim[ngames];
    for (i in 1:ngames) {
        home_s_sim[i] = poisson_rng(a[home_team[i]]*b[away_team[i]]*gamma);
        away_s_sim[i] = poisson_rng(a[away_team[i]]*b[home_team[i]]);
        home_st_sim[i] = binomial_rng(home_s_sim[i], f[home_team[i]]);
        away_st_sim[i] = binomial_rng(away_s_sim[i], f[away_team[i]]);
        home_goals_sim[i] = binomial_rng(home_st_sim[i], g[away_team[i]]);
        away_goals_sim[i] = binomial_rng(away_st_sim[i], g[home_team[i]]);
        }
}
