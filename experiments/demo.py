#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:26:12 2018

@author: goncalves1
"""
import sys
sys.path.append('..')
from design import Experiment
from datasets.UCIBreast import UCIBreast
from methods.IndependentMarginals import IndependentMarginals
from methods.MixtureProductMultinomials import MixtureProductMultinomials
from methods.MICE import MICE
from methods.CLGP import CLGP


def main():

    # create dataset and perform encoding
    dataset = UCIBreast()
    dataset.prepare_dataset()

    # CLGP hyper-parameters
    CLGP_inducing_points = 20
    CLGP_latent_space = 3

    # list of methods to compare against
    # note that the same method can be called many times just using
    # different hyper-parameter values
    methods = [IndependentMarginals(name='IM'),
               MixtureProductMultinomials(K=3, burn_in_steps=50, n_gibbs_steps=100, name='MPoM'),
               MICE('DecisionTree', name='MICE-DT'),
               MICE('LogisticRegression', name='MICE-LR'),
               CLGP(CLGP_inducing_points, CLGP_latent_space, lamb=0.01, reg=10, max_steps=30, name='CLGP'),
    ]

    # list of metrics to measure method's performance
    # see list of available metrics in utils/performance_metrics.py
    metrics = ['kl_divergence',
               'cross_classification',
            #    'cca_accuracy',
            #    'cluster_measure',
            #    'pairwise_correlation_difference',
               'membership_disclosure',
               'attribute_disclosure'
    ]

    # create an experiment and execute it
    exp_folder = __file__.strip('.py')
    exp = Experiment(exp_folder, dataset, methods,
                     metrics, nb_gens=2)
    exp.execute()
    exp.generate_report()


if __name__ == '__main__':

    main()
