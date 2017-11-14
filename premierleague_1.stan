data {
    int nteams;
    int ngames;
    int home_goals[ngames];
    int away_goals[ngames];
    int home_team[ngames];
    int away_team[ngames];
}
parameters {
    vector<lower=0>[nteams] a;
    vector<lower=0>[nteams] b;
    real<lower=0> gamma;
    real<lower=0> sigma_log_a;
    real<lower=0> sigma_log_b;
    real<lower=0> mu_log_b;
}
model {
    a ~ lognormal(0., sigma_log_a);
    b ~ lognormal(mu_log_b, sigma_log_b);
    gamma ~ lognormal(0, 1);
    sigma_log_a ~ normal(0, 10);
    sigma_log_b ~ normal(0, 10);
    mu_log_b ~ normal(0, 1);
    for (i in 1:ngames) {
        home_goals[i] ~ poisson(a[home_team[i]]*b[away_team[i]]*gamma);
        away_goals[i] ~ poisson(a[away_team[i]]*b[home_team[i]]);
        }
}
generated quantities {
    int<lower=0> home_goals_sim[ngames];
    int<lower=0> away_goals_sim[ngames];
    for (i in 1:ngames) {
        home_goals_sim[i] = poisson_rng(a[home_team[i]]*b[away_team[i]]*gamma);
        away_goals_sim[i] = poisson_rng(a[away_team[i]]*b[home_team[i]]);
        }
}
