# PFE_BayesianDeepNetworksForBigDataMining
Bayesian Deep Networks:
Bayesian Deep Learning - Images classification 
#
The objective of our project is to implement BAYESIAN DEEP NETWORKS by proposing an adequate approach (The approach of MONTE CARLO DROPOUT) to provide high predictive performance. To do this, we took advantage of the Deep Learning philosophy (Convolutive Neural Networks) to abstract observed characteristics and improve the classification of an image data set. In addition, we adopted the BAYESIAN NETWORK principle in order to make use of their uncertainty.
#
The approach of MONTE CARLO DROPOUT(Gal and Ghahramani, 2016):

The idea of this technique is to use the dropout during  the training phase and the test phase so that from a single input we can have several different output values through T MCD forward passes (a few hundred times), i.e. pass the same input to the network and then apply a random dropout. Intuitively, our model is able to give different predictions since different combinations of neurons are used for prediction. Moreover, this method is indeed Bayesian.
The manipulated instances are probability distributions and not scalars. Thus, the network weights and the values contained in neurons follow a gaussian probability (normal) that are characterized by an average value and a standard deviation. From these distributions, it is possible to calculate the different uncertainties using mathematical formulae. 
