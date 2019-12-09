# Fixed-effect Heterogenous Multinomial Probit
This project estimate a choice model between 3 alternatives: ethanol, regular gasoline, and midgrade gasoline.
The model accounts for (1) unobserved market factors with fuel-station fixed effects
(2) correlation in random utility between fuels, and (3) possibility of inaccurate price perception.
The model is estimate with a Heterogenous Multinomial Probit estimation with fuel-station fixed effects.

I use [Theano](http://deeplearning.net/software/theano/) for automatic differetiation, and 
[Ipopt](https://www.coin-or.org/Ipopt/documentation/) for non-linear optimization (maximum simulated likelihood)

This is part of my ongoing research with Prof. Alberto Salvo titled *"Price Salience and Imperfect Information: Evidences from Experiments at Fuelling Stations in Brazil"*
