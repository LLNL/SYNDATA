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

import sys
sys.path.append('..')

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from design import Dataset
from utils.DataFrameImputer import DataFrameImputer


class UCIBreast(Dataset):
    """
    Class storing UCI's Breast dataset. Data is encoded with Label Encoding,
    so that every category is mapped into an integer.

    Attributes:
            name (str): dataset name
    """
    def __init__(self, db_name='UCI-Breast'):
        """
        """
        self.db_name = db_name  # dataset identifier
        self.raw_data = None  # original data and categories
        self.enc_data = None  # data encoded into numerical values
        self.encoding_dict = defaultdict(LabelEncoder)

    def prepare_dataset(self):
        """
        TODO (goncalves1): docstrings
        """
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data'
        self.raw_data = pd.read_csv(url, sep=',', names=['class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
                                                     'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])

        # Label encoding: mapping categories to integers
        # It treats missing values `?` as another category, then the synthetic generator will
        # also sample records with missing values. If that's not desired, user can either remove
        # records with missing values, or impute them using her/his preferred technique.
        self.enc_data = self.raw_data.apply(lambda x: self.encoding_dict[x.name].fit_transform(x))

    def decode_data(self, data):
        # Map data back to original categories
        data.apply(lambda x: self.encoding_dict[x.name].inverse_transform(x))
        return data

