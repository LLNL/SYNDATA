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

import numpy as np
import pandas as pd
from design import Method


class IndependentMarginals(Method):
    """
    Concrete class implementing Independent Marginals.
    Attributes:
        name = string to be used as reference for the method
    """
    def __init__(self, name='IndependentMarginals'):
        """
            Validate input hyper-parameters and initialize class.
        """
        # initialize parameters
        super().__init__(name)
        self.min_prob = 1e-5
        self.output_directory = ''

    def fit(self, data):
        """ Estimate marginals distributions from the data. """
        self.column_names = data.columns
        self.marginal_dist = dict()
        self.logger.info('Estimating variables distribution.')
        for var in data.columns:
            freq_col = data[var].value_counts(normalize=True, sort=False)
            # make sure there is a minimum of chance of any element being
            # picked
            vals = np.maximum(self.min_prob, freq_col.values)
            vals = vals/vals.sum()
            self.logger.info(var)
            self.logger.info(vals)
            self.marginal_dist[var] = {'values': freq_col.index.tolist(),
                                       'p': vals}

    def generate_samples(self, nb_samples):
        """Generate samples from the independently estimated marginal dist. """
        synth_data = np.zeros((nb_samples, len(self.column_names)), dtype=int)
        for i, var in enumerate(self.column_names):
            arr = self.marginal_dist[var]['values']
            pbs = self.marginal_dist[var]['p']
            synth_data[:, i] = np.random.choice(arr, size=nb_samples, p=pbs)
        samples = pd.DataFrame(data=synth_data, columns=self.column_names)
        # print('IM: samples.dtypes: ', samples.dtypes)
        return samples

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
