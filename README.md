# FPL 

My playground for learning about predictive models of football matches, with a 
view to improving my performance on [Fantasy Premier League](https://fantasy.premierleague.com/). 

## Bayesian analysis in `stan`

To learn how to use `stan`, I implemented a heirarchical model that generates the number of shots on target and goals scored by each team in a match. 
The model is based on the [Dixon & Coles](http://www.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf), but extends it to include shots on target.
The hyperpriors on the team-level parameters are dependent on the pre-season predictions of the BBC journalist Phil McNulty.
