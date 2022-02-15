###############################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Andre Goncalves <andre@llnl.gov> and Priyadip Ray <ray34@llnl.gov>
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

import numpy as np
import pandas as pd
import pickle
from scipy.stats import dirichlet
from scipy.stats import beta
from design import Method


class MixtureProductMultinomials(Method):
    """
    Concrete class implementing the Product of Multinomials model.
    """
    def __init__(self, K, alpha=10, burn_in_steps=1000,
                 n_gibbs_steps=10000, name='MPoM'):
        """
            Validate input hyper-parameters and initialize class.
        """
        # initialize parameters
        super().__init__(name)
        assert isinstance(K, int), 'Truncation level K must be integer'
        self.K = K  # Truncation level of stick-breaking process
        self.alpha = alpha  # Parameter of Beta distribution
        self.burn_in_steps = burn_in_steps  # burn-in steps
        self.n_gibbs_steps = n_gibbs_steps  # maximum number of gibbs steps
        self.posterior = {'w': list(), 'p': list()}  # posterior samples

        self.output_directory = ''

    def fit(self, data, store_post_samples=False):
        """ Inference method: Gibbs sampler. """
        self.column_names = data.columns
        self.logger.info('Estimating variables distribution.')

        x = data.values
        nb_samples, dim = np.shape(data)  # data dimensionality

        # get the number levels for each categorical variable
        self.levels = [None]*len(self.column_names)
        for i, col in enumerate(self.column_names):
            self.levels[i] = len(data[col].unique())
            self.logger.info('{}: {} levels'.format(col, self.levels[i]))

        # Parameter for Dirichlet distribution
        d_p = [1.0/self.levels[i] for i in range(dim)]
        p, v, z, w = self.__init_variables(dim, nb_samples)

        for k in range(self.n_gibbs_steps):
            if k % 100 == 0:
                self.logger.info('Gibbs iteration: {} of {}'.format(k, self.n_gibbs_steps))
            v, w = self.__update_v(z, v, w)
            p = self.__update_p(p, z, d_p, x)
            z, _ = self.__update_z(p, z, w, d_p, x)

            if (k > self.burn_in_steps):
                # save posterios samples
                self.posterior['w'].append(w)
                self.posterior['p'].append(p)

        self.logger.info('Done with Gibbs inference')
        if store_post_samples:
            with open('posterior_samples.pkl', 'wb') as fh:
                pickle.dump(self.posterior, fh)

    def generate_samples(self, nb_samples):
        """Generate samples from the trained Product of Multinomials model. """
        nb_samples = np.minimum(nb_samples, len(self.posterior['w']))
        dim = len(self.column_names)
        synth_data = np.zeros((nb_samples, dim), dtype=int)
        for i in range(nb_samples):
            z_syn = np.argmax(np.random.multinomial(1, self.posterior['w'][i], 1))
            #  Draws from categorical distribution
            for k in range(dim):
                sample = np.random.multinomial(1, self.posterior['p'][i][k][z_syn], 1)
                synth_data[i, k] = np.argmax(sample)
        return pd.DataFrame(data=synth_data, columns=self.column_names)

    def __init_variables(self, dim, nb_samples):
        """ Initialize all parameters """
        p = {}
        for i in range(dim):
            p[i] = [[0 for j in range(self.levels[i])] for k in range(self.K)]
        v = [beta.rvs(1, self.alpha) for i in range(self.K)]
        w = np.zeros(self.K)
        w[0] = v[0]
        for j in range(self.K-2):
            prod = 1
            if (j > 0):
                for h in range(j):
                    prod = prod*(1-v[h])
                w[j] = v[j]*prod
        w[self.K-1] = 1 - np.sum(w[0:self.K-2])
        z = np.array([int(np.random.uniform(0, self.K))
                      for i in range(nb_samples)])
        return p, v, z, w

    def __update_v(self, z, v, w):
        """ Update v-variable for stick-breaking process """
        for k in range(self.K):
            temp_1 = np.where(z == k)[0]
            temp_2 = np.shape(temp_1)[0]
            temp_sum = 0
            for j in range(k+1, self.K+1):
                temp_3 = np.where(z == j)[0]
                temp_4 = np.shape(temp_3)[0]
                temp_sum = temp_sum + temp_4
            # Check for correct beta parameters
            v[k] = beta.rvs(1 + temp_2, self.alpha + temp_sum)

        prod = 1
        w[0] = v[0]
        for i in range(self.K-1):
            prod = 1
            if (i > 0):
                for h in range(i):
                    prod = prod*(1-v[h])
                w[i] = v[i]*prod
        w[self.K-1] = 1 - np.sum(w[0:self.K-1])
        return v, w

    def __update_p(self, p, z, d_p, x):
        """ Update p-variable probability for each component (weights) """
        nb_samples, dim = x.shape
        for j in range(dim):  # for each feature
            for k in range(self.K):  # for each cluster
                updated_param = np.zeros(self.levels[j])
                temp = np.where(z == k)[0]
                t_array = x[temp]
                t_array_1 = t_array[:, j]
                for r in range(self.levels[j]):
                    temp_2 = np.shape(np.where(t_array_1 == r)[0])[0]
                    updated_param[r] = d_p[j] + temp_2
                p[j][k] = dirichlet.rvs(updated_param, 1)[0]
        return p

    def __update_z(self, p, z, w, d_p, x):
        """ Update z-variable cluster label """
        nb_samples, dim = x.shape
        for i in range(nb_samples):
            prob = np.zeros(self.K)
            for l in range(self.K):
                s_t = 0
                for j in range(dim):
                    v_t = x[i, j]
                    s_t = s_t + np.log(p[j][l][v_t])
                prob[l] = s_t + np.log(w[l])
                prob[l] = np.exp(prob[l])
            prob = prob/np.sum(prob)
            if(np.shape(np.where(np.isnan(prob))[0])[0] > 0):
                prob = 1.0/self.K + np.zeros(self.K)
            z[i] = np.argmax(np.random.multinomial(1, prob, 1))
        return z, prob

    # def set_output_directory(self, output_dir):
    #     """ Set output folder path.
    #     Args:
    #         output_dir (str): path to output directory.
    #     """
    #     self.output_directory = output_dir
    #     self.logger.set_path(output_dir)
    #     self.logger.setup_logger(self.__str__())

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
