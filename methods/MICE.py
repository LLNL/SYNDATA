###############################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Andre Goncalves <andre@llnl.gov>
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from design import Method

CLASSIFIERS =  ('LogisticRegression', 'DecisionTree', 'RandomForest')


class MICE(Method):
    """
    Concrete class implementing MICE (Multiple Imputation by Chained Equations).
    Attributes:
        name = string to be used as reference for the method
    """

    def __init__(self, method='LogisticRegression',
                 order='ascending', joint_modeled_cats=1,
                 name='MICE'):
        """
            Validate input hyper-parameters and initialize class.
        """
        # initialize parameters
        super().__init__(name)
        assert method in CLASSIFIERS, "Unknown classifier."
        self.method = method
        self.order = order
        self.joint_modeled_cats = 1  # joint_modeled_cats
        self.classifiers = list()
        self.joint_pmf = None
        self.min_prob = 0

    def fit(self, data):
        """ Estimate marginals distributions from the data. """
        self.logger.info('Estimating variables distribution.')

        self.column_names = data.columns
        nb_cats = np.array([len(data[col].unique()) for col in data.columns])

        if self.order.startswith('descending'):
            sorted_cats_ids = np.argsort(-nb_cats)
        elif self.order.startswith('ascending'):
            sorted_cats_ids = np.argsort(nb_cats)
        else:
            raise ValueError('Unknown order: {}'.format(self.order))

        self.sorted_cats_ids = sorted_cats_ids.copy()

        self._estimate_joint_pmf(data, sorted_cats_ids)

        data = data.values.copy()  # convert from pandas.df to np array
        if self.method == 'LogisticRegression':
            clf = LogisticRegression()

        k = self.joint_modeled_cats
        for i, t in enumerate(sorted_cats_ids[self.joint_modeled_cats:]):
            if np.unique(data[:, t]).shape[0] > 1:
                if self.method == 'LogisticRegression':
                    self.classifiers.append(LogisticRegression())
                elif self.method == 'DecisionTree':
                    self.classifiers.append(DecisionTreeClassifier())
                elif self.method == 'RandomForest':
                    self.classifiers.append(RandomForestClassifier(n_estimators=100, max_depth=3))
            else:
                self.classifiers.append(ConstantPrediction())

            self.classifiers[i].fit(data[:, sorted_cats_ids[:k]], data[:, t])
            k += 1
        self.logger.info('Training completed.')

    def generate_samples(self, nb_samples):
        """Generate samples from the independently estimated marginal dist. """
        synth_data = np.zeros((nb_samples, len(self.column_names)), dtype=int)

        # generate first column from
        synth_data[:, 0] = np.random.choice(self.joint_pmf['values'],
                                            size=nb_samples,
                                            p=self.joint_pmf['p'])
        nb_vars_to_sample = len(self.column_names) - self.joint_modeled_cats

        for i in range(nb_vars_to_sample):
            k = i + self.joint_modeled_cats
            ps = self.classifiers[i].predict_proba(synth_data[:, 0:k])
            for j in range(nb_samples):
                synth_data[j, k] = np.random.choice(self.classifiers[i].classes_,
                                                    size=1,
                                                    p=ps[j, :])

        synth_data = synth_data[:, np.argsort(self.sorted_cats_ids)]
        df = pd.DataFrame(data=synth_data, columns=self.column_names)
        return df

    def _estimate_joint_pmf(self, data, sorted_ids):
        freq_col = data[self.column_names[sorted_ids[0]]].value_counts(normalize=True, sort=False)
        # make sure there is a minimum of chance of any element being picked
        vals = np.maximum(self.min_prob, freq_col.values)
        vals = vals / vals.sum()
        self.joint_pmf = {'values': freq_col.index.tolist(), 'p': vals}

    def set_params(self, **kwargs):
        """
        Set method's hyper-parameters.
        Args
            No args.
        """
        pass

    def get_params(self):
        """
        Return defined hyper-parameters. (no hyper-parameters here)
        """
        return None


class ConstantPrediction(object):
    """ Dummy classifier that just predicts the same value all the time.
        This is used in the case the y variable has only one possible value
        (single class classification problem). """

    def __init__(self):
        self.classes_ = None  # vector of possible class values

    def fit(self, _x, y):
        self.classes_ = np.zeros((1, )) + np.unique(y)[0]

    def predict_proba(self, x):
        return np.ones((x.shape[0], 1))