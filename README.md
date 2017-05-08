# FPL

Testing branch. 

My playground for learning about predictive models of football matches, with a 
view to improving my performance on [Fantasy Premier League](https://fantasy.premierleague.com/). 

## The [Dixon & Coles](http://www.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf) model

I implemented the Dixon & Coles (1997) model and applied it to the 2015/16 Premier League season in a Jupyter notebook, which you should view on [nbviewer](https://nbviewer.jupyter.org/github/anguswilliams91/FPL/blob/master/DixonColesAnalysis.ipynb), so that you can toggle the code on and off.
My `python` implementation of the model is found in `dixoncoles.py`.

I also created a new model, based on Dixon & Coles, that predicts the number of shots on target as well as the number of goals.
I wrote a [notebook about that](https://nbviewer.jupyter.org/github/anguswilliams91/FPL/blob/master/ShotsOnTarget.ipynb), too. 
Again, the implementation is found in `dixoncoles.py`.
