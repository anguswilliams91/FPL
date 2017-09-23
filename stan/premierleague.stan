data {
    int nteams;
    int ngames;
    int home_goals[ngames];
    int away_goals[ngames];
    int home_st[ngames];
    int away_st[ngames];
    int home_team[ngames];
    int away_team[ngames];
    int mcnulty_rank[nteams];
}
transformed data {
    real normalised_rank[nteams];
    int home_total_st[ngames];
    int away_total_st[ngames];
    for (i in 1:nteams) normalised_rank[i] = 2*((1. - mcnulty_rank[i])/19. + 0.5);
    for (i in 1:ngames) home_total_st[i] = home_st[i] + home_goals[i];
    for (i in 1:ngames) away_total_st[i] = away_st[i] + away_goals[i];
}
parameters {
    real<lower=0> a[nteams];
    real<lower=0> b[nteams];
    real<lower=0, upper=1> f[nteams];
    real<lower=0> gamma;
    real corr_a;
    real corr_b;
    real corr_f;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma_f;
    real<lower=0> mu_b;
    real<lower=0, upper=1> mu_f;
}
model {
    for (i in 1:nteams){
        a[i] ~ normal(1 + corr_a*normalised_rank[i], sigma_a);
        b[i] ~ normal(mu_b + corr_b*normalised_rank[i], sigma_b);
        f[i] ~ normal(mu_f + corr_f*normalised_rank[i], sigma_f);
        }
    gamma ~ normal(1.1, 0.3);
    corr_a ~ normal(0, 1);
    corr_b ~ normal(0, 1);
    sigma_a ~ normal(0, 10);
    sigma_b ~ normal(0, 10);
    sigma_f ~ normal(0, 1);
    mu_b ~ normal(0, 10);
    mu_f ~ normal(.5, .5);
    for (i in 1:ngames) {
        home_total_st[i] ~ poisson(a[home_team[i]]*b[away_team[i]]*gamma);
        away_total_st[i] ~ poisson(a[away_team[i]]*b[home_team[i]]);
        home_goals[i] ~ binomial(home_total_st[i], f[home_team[i]]);
        away_goals[i] ~ binomial(away_total_st[i], f[away_team[i]]);
        }
}
generated quantities {
    int<lower=0> home_goals_sim[ngames];
    int<lower=0> away_goals_sim[ngames];
    int<lower=0> home_st_sim[ngames];
    int<lower=0> away_st_sim[ngames];
    for (i in 1:ngames) {
        home_st_sim[i] = poisson_rng(a[home_team[i]]*b[away_team[i]]*gamma);
        away_st_sim[i] = poisson_rng(a[away_team[i]]*b[home_team[i]]);
        home_goals_sim[i] = binomial_rng(home_st_sim[i], f[home_team[i]]);
        away_goals_sim[i] = binomial_rng(away_st_sim[i], f[away_team[i]]);
        }
}