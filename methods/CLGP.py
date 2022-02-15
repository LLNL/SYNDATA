###############################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Rui Meng <meng1@ucsc.edu> and Andre Goncalves <andre@llnl.gov>
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
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
from design import Method


class CLGP(Method):
    """
    Concrete class implementing CLGP model in tensorflow from scratch.
    Reference paper: http://proceedings.mlr.press/v37/gala15.pdf
    """
    def __init__(self, M, Q, T=5, lamb=0.01,
                 reg=100, max_steps=500, learning_rate=0.02,
                 error=1e-5, verbose=False, name='CLGP'):
        """
            Validate input hyper-parameters and initialize class.
        """
        super().__init__(name)
        self.M = M  # number of inducing points
        self.Q = Q  # number of latent dimensions
        self.T = T  # number of samples for Monte Carlo integration (Eq. 11)
        self.reg = reg
        self.max_steps = max_steps  # number of maximum optimization steps
        self.learning_rate = learning_rate  # learning rate for Adam
        self.error = error # to avoid numerical issues
        self.lamb_val = lamb
        self.nb_training_samples = -1
        self.name = name
        self.verbose = verbose
        self.model_path = ''

        self.sess = tf.compat.v1.Session()

    def fit(self, data):

        self.column_names = data.columns

        N, D = data.shape
        self.nb_training_samples = N

        var_levels = np.zeros(data.shape[1], dtype=np.int)
        for indx, col in enumerate(self.column_names):
            var_levels[indx] = len(data[col].unique())
        self.logger.info("The categorical levels are {}".format(var_levels))

        self.create_graph(var_levels, N)
        self.optimize(data)

    def create_graph(self, var_levels, sample_size):

        self.var_levels = var_levels
        K_max = max(self.var_levels)
        K = self.var_levels
        D = len(self.var_levels)
        N = sample_size

        # Define variables
        with tf.compat.v1.variable_scope('network_build') as net_build:
            self.theta = tf.Variable(tf.zeros([D, self.Q + 2],
                                         dtype=tf.float32), name="theta")
            self.Z = tf.Variable(np.random.randn(self.M, self.Q),
                            dtype=tf.float32, name="Z")
            self.mu = tf.Variable(np.random.randn(self.M, D, K_max)*0.01,
                             dtype=tf.float32, name="mu")
            self.L = tf.Variable(np.random.randn(D, self.M*(self.M+1)//2),
                            dtype=tf.float32, name="L")
            self.m = tf.Variable(tf.zeros([N, self.Q], dtype=tf.float32), name="m")
            self.s = tf.Variable(np.log(np.ones([N, self.Q])*0.1),
                            dtype=tf.float32, name="s")
            self.lamb = tf.compat.v1.placeholder(tf.float32, name='lamb')

        with tf.compat.v1.variable_scope('input_data') as input_data:
            self.y = tf.compat.v1.placeholder(tf.int32, shape=(sample_size, D), name='y')

        # Define hyper-parameters
        sigma_x = 1  # scale prior for X

        with tf.name_scope('Dist_X'):
            QX = tfp.distributions.Normal(loc=self.m, scale=tf.exp(self.s), name="QX")
            PX = tfp.distributions.Normal(loc=np.zeros([N, self.Q]).astype(np.float32), 
                                          scale=sigma_x*np.ones([N, self.Q]).astype(np.float32), name="PX")

        Ld_list = []
        cov_mat_list = []
        for d in range(D):
            Ld = self.L[d, :]
            Ld_list.append(self._vector_to_tril_mat(Ld))
            cov_mat_list.append(self._covariance_function(self.theta[d,:], self.Z))

        with tf.name_scope('Dist_U'):
            QU_loc = [[self.mu[:, d, k] for k in range(K_max)] for d in range(D)]
            eyes = tf.eye(self.M)
            QU_scale_tril = [[Ld_list[d] if k < K[d] else eyes for k in range(K_max)] for d in range(D)]
            QU = tfd.Independent(distribution=tfd.MultivariateNormalTriL(loc = QU_loc, scale_tril=QU_scale_tril), 
                                 reinterpreted_batch_ndims=0, name = "QU")
            PU_loc = [[tf.zeros(self.M) for k in range(K_max)] for d in range(D)]
            PU_covariance_matrix = [[cov_mat_list[d] if k < K[d] else eyes for k in range(K_max)] for d in range(D)]
            PU = tfd.Independent(distribution=tfd.MultivariateNormalFullCovariance(loc=PU_loc, covariance_matrix=PU_covariance_matrix), 
                                 reinterpreted_batch_ndims=0, name = "PU")

        # Define KL divergence on X: KL(q(X)||p(X))
        with tf.name_scope('KL_X'):
            self.KL_X = tf.reduce_sum(tfp.distributions.kl_divergence(QX, PX, name='KL_X'))

        # Define KL divergence on U: KL(q(U)|p(U))
        with tf.name_scope('KL_U'):
            KL_U_mat = tfp.distributions.kl_divergence(QU, PU, name='KL_U_mat')
            indx = [[d, k] for d in range(D) for k in range(K[d])]
            self.KL_U = tf.reduce_sum(tf.gather_nd(KL_U_mat, indx), name='KL_U')
            # tf.summary.scalar("summary/KL_U", KL_U)

        with tf.name_scope('KL_ZX'):
            # estimate distribution of all Zs using gaussian distribution
            Q_Z_loc = tf.reduce_mean(self.Z, axis=0)
            Q_Z_cov = self._tf_cov(self.Z)
            Q_Z = tfd.MultivariateNormalFullCovariance(loc=Q_Z_loc, 
                                                       covariance_matrix=Q_Z_cov,
                                                       name='ED_Z')
            # estimate distribution of all Xs for each time using gaussian distribution
            m_mat = tf.reshape(self.m, [-1, self.Q])
            Q_X_loc = tf.reduce_mean(m_mat, axis=0)
            Q_X_cov = self._tf_cov(m_mat)
            Q_X = tfd.MultivariateNormalFullCovariance(loc=Q_X_loc, 
                                                       covariance_matrix=Q_X_cov,
                                                       name='ED_X')
            # compute the KL divergence between X and Z
            self.KL_ZX = tfp.distributions.kl_divergence(Q_Z, Q_X,
                                                         name='KL_ZX')

        with tf.name_scope('Comp_F'):
            Comp_F = tf.constant(0, dtype=tf.float32, name="Comp_F")
            np_array = tf.expand_dims(tf.range(sample_size, dtype=tf.int32), axis=1)
            for t in range(self.T):
                sampled_eps_X = tf.random.normal([N, self.Q])
                sampled_X = self.m + tf.multiply(tf.exp(self.s), sampled_eps_X)
                Comp_F_t = 0
                # print("QX has been sampled.")
                sampled_U = []
                for d in range(D):
                    sampled_eps_U_d = tf.random.normal([self.M, K[d]])
                    sampled_U_d = self.mu[:, d, :K[d]] + tf.matmul(Ld_list[d], sampled_eps_U_d)
                    sampled_U.append(sampled_U_d)

                a_d_list = []
                b_d_list = []
                for d in range(D):
                    Inv_cov_d_MM = tf.linalg.inv(self._covariance_function(self.theta[d,:], self.Z))
                    cov_d_MN = self._covariance_function(self.theta[d,:], self.Z, sampled_X)
                    cov_d_NM = tf.transpose(cov_d_MN)
                    # a_d: M by N
                    a_d_list.append(tf.matmul(Inv_cov_d_MM, cov_d_MN))
                    # B_d: N
                    B_d = tf.reshape(tf.reduce_sum(tf.multiply(cov_d_NM, tf.transpose(a_d_list[d])), axis=1), [-1])
                    # b_d: N
                    b_d = (tf.exp(self.theta[d,0]) + tf.exp(self.theta[d,1]))*tf.constant(np.ones(N), dtype=np.float32) - B_d
                    zeros = tf.zeros_like(b_d)
                    masked = b_d > 0
                    b_d = tf.where(masked, b_d, zeros)
                    b_d_list.append(b_d)

                for d in range(D):
                    # t_s_qfd = time.time()
                    sampled_eps_f_d = tf.random.normal([N, K[d]])
                    sampled_f_d = tf.matmul(tf.transpose(a_d_list[d]), sampled_U[d]) + tf.multiply(tf.tile(tf.reshape(tf.sqrt(b_d_list[d]), [-1,1]), [1,K[d]]), sampled_eps_f_d)
                    y_d_indx = tf.concat((np_array, self.y[:, d, None]), axis=1)

                    Comp_F_t += tf.reduce_sum(tf.math.log(tf.gather_nd(tf.nn.softmax(sampled_f_d), y_d_indx)))

                Comp_F += Comp_F_t
            self.Comp_F = Comp_F/self.T

        with tf.name_scope('model'):
            self.elbo = -self.lamb*(self.KL_X + self.KL_U + self.reg*self.KL_ZX) + self.Comp_F

        with tf.name_scope('generative_model'):
            self.new_Y = []
            for d in range(D):
                sampled_eps_f_d = tf.random.normal([N, K[d]])
                sampled_f_d = tf.matmul(tf.transpose(a_d_list[d]), sampled_U[d]) + tf.multiply(tf.tile(tf.reshape(tf.sqrt(b_d_list[d]), [-1,1]), [1,K[d]]), sampled_eps_f_d)
                sampled_y_d = tf.random.categorical(tf.math.log(tf.nn.softmax(sampled_f_d)), 1)
                self.new_Y.append(sampled_y_d)
            self.new_Y = tf.concat(self.new_Y, axis=1)

        with tf.compat.v1.variable_scope('network_train') as net_train:
            with tf.name_scope('train'):
                opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
                self.train = opt.minimize(-self.elbo)  # negative of ELBO

        init_all_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_all_op)

    def optimize(self, data):
        """ Estimate marginals distributions from the data. """

        # store column names for later use
        self.column_names = data.columns

        N, D = data.shape
        self.nb_training_samples = N
       
        samples = data.astype(np.int32)

        if self.verbose:
            print('ELBO\tComp_F0\tKL_U\tKL_X')

        lamb=self.lamb_val
        for step in range(self.max_steps):
            self.sess.run(self.train, feed_dict={self.lamb: lamb, self.y: samples})
            lamb = min(1.0, lamb+0.01)
            elbo0, theta0, Z0, KL_U0, KL_X0, KL_ZX0, Comp_F0 = self.sess.run([self.elbo, self.theta, 
                                                                              self.Z, self.KL_U, self.KL_X, 
                                                                              self.KL_ZX, self.Comp_F], 
                                                                              feed_dict={self.lamb: lamb, self.y: samples})
            if self.verbose:
                print('[step {}]: ELBO={}'.format(step, elbo0))            

    def generate_samples(self, nb_samples):
        """Generate samples from the independently estimated marginal dist. """
        synth_data = list()
        count = 0
        while count < nb_samples:
            n_i = np.minimum(nb_samples-count, 
                             self.nb_training_samples)
            synth_data_i = self.sess.run(self.new_Y)
            if n_i < synth_data_i.shape[0]:
                idx = np.random.choice(np.arange(synth_data_i.shape[0]), n_i)
                synth_data_i = synth_data_i[idx, :]
            synth_data.append(synth_data_i)
            count += n_i
        samples = np.concatenate(synth_data)
        samples = pd.DataFrame(data=samples,
                               columns=self.column_names)
        return samples

    def _tf_cov(self, x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        cov_xx += np.diag(np.ones(cov_xx.shape[0]))*self.error
        return cov_xx

    def _vector_to_tril_mat(self, L):
        idx = np.zeros([self.M, self.M], dtype=np.int32)
        mask = np.zeros([self.M, self.M], dtype=np.bool)
        tril_idx = np.tril_indices(self.M)
        idx[tril_idx] = np.arange((self.M*(self.M+1)/2))
        mask[tril_idx] = True
        Ld = tf.where(mask, tf.gather(L, idx),
                      tf.zeros([self.M, self.M], dtype=L.dtype))
        return Ld

    def _covariance_function(self, theta, X1, X2 = None):
        # theta = (alpha, Lambda)
        sigmaf2= tf.exp(theta[0])
        sigmaf2_noise = tf.exp(theta[1])
        l = tf.exp(theta[-self.Q:])
        _X2 = X1 if X2 is None else X2
        if len(X1.shape) == 1:
            X1 = tf.reshape(X1, [1, -1])
        if len(_X2.shape) == 1:
            _X2 = tf.reshape(_X2, [1, -1])
        dist = tf.matmul(tf.reshape(tf.reduce_sum((X1/l)**2,1), [-1,1]), tf.reshape(tf.ones(_X2.shape[0]), [1,-1])) + tf.matmul(tf.reshape(tf.ones(X1.shape[0]), [-1,1]), tf.reshape(tf.reduce_sum((_X2/l)**2,1), [1,-1])) - 2*tf.matmul((X1/l), tf.transpose(_X2/l))
        cov_mat = sigmaf2 * tf.exp(-dist/2.0)
        return cov_mat+tf.eye(X1.shape.as_list()[0])*sigmaf2_noise+np.diag(np.ones(X1.shape.as_list()[0])*self.error) if X2 is None else cov_mat

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
