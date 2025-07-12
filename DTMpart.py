from gensim import utils, matutils
from gensim.models import ldamodel
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
import logging
from six.moves import range, zip
import pandas as pd
import random
import math
from gensim.corpora import Dictionary
import json

logger = logging.getLogger(__name__)

class sslm(utils.SaveLoad):
    """Encapsulate the inner State Space Language Model for DTM.

    Some important attributes of this class:

        * `obs` is a matrix containing the document to topic ratios.
        * `e_log_prob` is a matrix containing the topic to word ratios.
        * `mean` contains the mean values to be used for inference for each word for a time slice.
        * `variance` contains the variance values to be used for inference of word in a time slice.
        * `fwd_mean` and`fwd_variance` are the forward posterior values for the mean and the variance.
        * `zeta` is an extra variational parameter with a value for each time slice.

    """
    """
    为DTM封装内部状态空间语言模型。

     此类的一些重要属性：

         *`obs`是一个包含文档与主题比率的矩阵。
         *`e_log_prob`是一个包含主题与单词比率的矩阵。
         *`mean`包含一个时间片中每个单词的推论平均值。
         *`variance`包含用于在时间片中推断单词的方差值。
         * fwd_mean和fwd_variance是均值和方差的正后验值。
         *`zeta`是一个额外的变分参数，每个时间片都有一个值。
    """
    def __init__(self, vocab_len=None, num_time_slices=None, num_topics=None, obs_variance=0.5, chain_variance=0.005):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.num_topics = num_topics

        # setting up matrices
        self.obs = np.zeros((vocab_len, num_time_slices))
        self.e_log_prob = np.zeros((vocab_len, num_time_slices))
        self.mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_variance = np.zeros((vocab_len, num_time_slices + 1))
        self.variance = np.zeros((vocab_len, num_time_slices + 1))
        self.zeta = np.zeros(num_time_slices)

        # the following are class variables which are to be integrated during Document Influence Model
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

    def update_zeta(self):
        """Update the Zeta variational parameter.

        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),
        over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.

        Returns
        -------
        list of float
            The updated zeta values for each time slice.

        """
        for j, val in enumerate(self.zeta):
            self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta

    def compute_post_variance(self, word, chain_variance):
        r"""Get the variance, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        This function accepts the word to compute variance for, along with the associated sslm class object,
        and returns the `variance` and the posterior approximation `fwd_variance`.

        Notes
        -----
        This function essentially computes Var[\beta_{t,w}] for t = 1:T

        .. :math::

            fwd\_variance[t] \equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\ for\ 1:t) =
            (obs\_variance / fwd\_variance[t - 1] + chain\_variance + obs\_variance ) *
            (fwd\_variance[t - 1] + obs\_variance)

        .. :math::

            variance[t] \equiv E((beta_{t,w}-mean\_cap_{t,w})^2 |beta\_cap_{t}\ for\ 1:t) =
            fwd\_variance[t - 1] + (fwd\_variance[t - 1] / fwd\_variance[t - 1] + obs\_variance)^2 *
            (variance[t - 1] - (fwd\_variance[t-1] + obs\_variance))

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the variance of each word in each time slice, the second value is the
            inferred posterior variance for the same pairs.

        """
        INIT_VARIANCE_CONST = 1000

        T = self.num_time_slices
        variance = self.variance[word]
        fwd_variance = self.fwd_variance[word]
        # forward pass. Set initial variance very high
        fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST
        for t in range(1, T + 1):
            if self.obs_variance:
                c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            else:
                c = 0
            fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)

        # backward pass
        variance[T] = fwd_variance[T]
        for t in range(T - 1, -1, -1):
            if fwd_variance[t] > 0.0:
                c = np.power((fwd_variance[t] / (fwd_variance[t] + chain_variance)), 2)
            else:
                c = 0
            variance[t] = (c * (variance[t + 1] - chain_variance)) + ((1 - c) * fwd_variance[t])

        return variance, fwd_variance

    def compute_post_mean(self, word, chain_variance):
        """Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        Notes
        -----
        This function essentially computes E[\beta_{t,w}] for t = 1:T.

        .. :math::

            Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
            = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +
            (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

        .. :math::

            Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
            = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +
            (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the mean of each word in each time slice, the second value is the
            inferred posterior mean for the same pairs.

        """
        T = self.num_time_slices
        obs = self.obs[word]
        fwd_variance = self.fwd_variance[word]
        mean = self.mean[word]
        fwd_mean = self.fwd_mean[word]

        # forward
        fwd_mean[0] = 0
        for t in range(1, T + 1):
            c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]

        # backward pass
        mean[T] = fwd_mean[T]
        for t in range(T - 1, -1, -1):
            if chain_variance == 0.0:
                c = 0.0
            else:
                c = chain_variance / (fwd_variance[t] + chain_variance)
            mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
        return mean, fwd_mean

    def compute_expected_log_prob(self):
        """Compute the expected log probability given values of m.

        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
        The below implementation is the result of solving the equation and is implemented as in the original
        Blei DTM code.

        Returns
        -------
        numpy.ndarray of float
            The expected value for the log probabilities for each word and time slice.

        """
        for (w, t), val in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        return self.e_log_prob

    def sslm_counts_init(self, obs_variance, chain_variance, sstats):
        """Initialize the State Space Language Model with LDA sufficient statistics.

        Called for each topic-chain and initializes initial mean, variance and Topic-Word probabilities
        for the first time-slice.

        Parameters
        ----------
        obs_variance : float, optional
            Observed variance used to approximate the true and forward variance.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
        sstats : numpy.ndarray
            Sufficient statistics of the LDA model. Corresponds to matrix beta in the linked paper for time slice 0,
            expected shape (`self.vocab_len`, `num_topics`).

        """
        W = self.vocab_len
        T = self.num_time_slices

        log_norm_counts = np.copy(sstats)
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)

        # setting variational observations to transformed counts
        self.obs = (np.repeat(log_norm_counts, T, axis=0)).reshape(W, T)
        # set variational parameters
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance

        # compute post variance, mean
        for w in range(W):
            self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.chain_variance)
            self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.chain_variance)

        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()

    def fit_sslm(self, sstats):
        """Fits variational distribution.

        This is essentially the m-step.
        Maximizes the approximation of the true posterior for a particular topic using the provided sufficient
        statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and
        :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
            current time slice, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The lower bound for the true posterior achieved using the fitted approximate distribution.

        """
        W = self.vocab_len
        bound = 0
        old_bound = 0
        sslm_fit_threshold = 1e-6
        sslm_max_iter = 2
        converged = sslm_fit_threshold + 1

        # computing variance, fwd_variance
        self.variance, self.fwd_variance = \
            (np.array(x) for x in zip(*(self.compute_post_variance(w, self.chain_variance) for w in range(W))))

        # column sum of sstats
        totals = sstats.sum(axis=0)
        iter_ = 0

        model = "DTM"
        if model == "DTM":
            bound = self.compute_bound(sstats, totals)
        if model == "DIM":
            bound = self.compute_bound_fixed(sstats, totals)

        logger.info("initial sslm bound is %f", bound)

        while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
            iter_ += 1
            old_bound = bound
            self.obs, self.zeta = self.update_obs(sstats, totals)
            #obs = self.obs#change to get obs

            if model == "DTM":
                bound = self.compute_bound(sstats, totals)
            if model == "DIM":
                bound = self.compute_bound_fixed(sstats, totals)

            converged = np.fabs((bound - old_bound) / old_bound)
            logger.info("iteration %i iteration lda seq bound is %f convergence is %f", iter_, bound, converged)

        self.e_log_prob = self.compute_expected_log_prob()
        return bound

    def compute_bound(self, sstats, totals):
        """Compute the maximized lower bound achieved for the log probability of the true posterior.

        Uses the formula presented in the appendix of the DTM paper (formula no. 5).

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        float
            The maximized lower bound.

        """
        w = self.vocab_len
        t = self.num_time_slices

        term_1 = 0
        term_2 = 0
        term_3 = 0

        val = 0
        ent = 0

        chain_variance = self.chain_variance
        # computing mean, fwd_mean
        self.mean, self.fwd_mean = \
            (np.array(x) for x in zip(*(self.compute_post_mean(w, self.chain_variance) for w in range(w))))
        self.zeta = self.update_zeta()

        val = sum(self.variance[w][0] - self.variance[w][t] for w in range(w)) / 2 * chain_variance

        #logger.info("Computing bound, all times")

        for t in range(1, t + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(w):

                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]

                v = self.variance[w][t]

                # w_phi_l is only used in Document Influence Model; the values are always zero in this case
                # w_phi_l = sslm.w_phi_l[w][t - 1]
                # exp_i = np.exp(-prev_m)
                # term_1 += (np.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) -
                # (v / chain_variance) - np.log(chain_variance)

                term_1 += \
                    (np.power(m - prev_m, 2) / (2 * chain_variance)) - (v / chain_variance) - np.log(chain_variance)
                term_2 += sstats[w][t - 1] * m
                ent += np.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1

        return val

    def update_obs(self, sstats, totals):
        """Optimize the bound with respect to the observed variables.

        TODO:
        This is by far the slowest function in the whole algorithm.
        Replacing or improving the performance of this would greatly speed things up.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
            对应于相应文档中的第一个时间片下的beta分布
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        (numpy.ndarray of float, numpy.ndarray of float)
            The updated optimized values for obs and the zeta variational parameter.

        """

        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 1e-3

        W = self.vocab_len
        T = self.num_time_slices

        runs = 0
        mean_deriv_mtx = np.zeros((T, T + 1))

        norm_cutoff_obs = None
        for w in range(W):
            w_counts = sstats[w]
            counts_norm = 0
            # now we find L2 norm of w_counts
            for i in range(len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]

            counts_norm = np.sqrt(counts_norm)

            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))

                # TODO: apply lambda function
                for t in range(T):
                    mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])

                deriv = np.zeros(T)
                args = self, w_counts, totals, mean_deriv_mtx, w, deriv
                obs = self.obs[w]
                model = "DTM"

                if model == "DTM":
                    # slowest part of method
                    obs = optimize.fmin_cg(
                        f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0
                    )
                if model == "DIM":
                    pass
                runs += 1

                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs

                self.obs[w] = obs

        self.zeta = self.update_zeta()

        return self.obs, self.zeta

    def compute_mean_deriv(self, word, time, deriv):
        """Helper functions for optimizing a function.

        Compute the derivative of:

        .. :math::

            E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.

        Parameters
        ----------
        word : int
            The word's ID.
        time : int
            The time slice.
        deriv : list of float
            Derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """

        T = self.num_time_slices
        fwd_variance = self.variance[word]

        deriv[0] = 0

        # forward pass
        for t in range(1, T + 1):
            if self.obs_variance > 0.0:
                w = self.obs_variance / (fwd_variance[t - 1] + self.chain_variance + self.obs_variance)
            else:
                w = 0.0
            val = w * deriv[t - 1]
            if time == t - 1:
                val += (1 - w)
            deriv[t] = val

        for t in range(T - 1, -1, -1):
            if self.chain_variance == 0.0:
                w = 0.0
            else:
                w = self.chain_variance / (fwd_variance[t] + self.chain_variance)
            deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]

        return deriv

    def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):
        """Derivation of obs which is used in derivative function `df_obs` while optimizing.

        Parameters
        ----------
        word : int
            The word's ID.
        word_counts : list of int
            Total word counts for each time slice.
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.
        mean_deriv_mtx : list of float
            Mean derivative for each time slice.
        deriv : list of float
            Mean derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """

        # flag
        init_mult = 1000

        T = self.num_time_slices

        mean = self.mean[word]
        variance = self.variance[word]

        # only used for DIM mode
        # w_phi_l = self.w_phi_l[word]
        # m_update_coeff = self.m_update_coeff[word]

        # temp_vector holds temporary zeta values
        self.temp_vect = np.zeros(T)

        for u in range(T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)

        for t in range(T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0

            for u in range(1, T + 1):
                mean_u = mean[u]
                mean_u_prev = mean[u - 1]
                dmean_u = mean_deriv[u]
                dmean_u_prev = mean_deriv[u - 1]

                term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
                term2 += (word_counts[u - 1] - (totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1])) * dmean_u

                model = "DTM"
                if model == "DIM":
                    # do some stuff
                    pass

            if self.chain_variance:
                term1 = - (term1 / self.chain_variance)
                term1 = term1 - (mean[0] * mean_deriv[0]) / (init_mult * self.chain_variance)
            else:
                term1 = 0.0

            deriv[t] = term1 + term2 + term3 + term4

        return deriv
def f_obs(x, *args):
    """Function which we are optimising for minimizing obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The value of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args
    # flag
    init_mult = 1000

    T = len(x)
    val = 0
    term1 = 0
    term2 = 0

    # term 3 and 4 for DIM
    term3 = 0
    term4 = 0

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    mean = sslm.mean[word]
    variance = sslm.variance[word]

    # only used for DIM mode
    # w_phi_l = sslm.w_phi_l[word]
    # m_update_coeff = sslm.m_update_coeff[word]

    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]

        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * np.exp(mean_t + variance[t] / 2) / sslm.zeta[t - 1]

        model = "DTM"
        if model == "DIM":
            # stuff happens
            pass

    if sslm.chain_variance > 0.0:

        term1 = - (term1 / (2 * sslm.chain_variance))
        term1 = term1 - mean[0] * mean[0] / (2 * init_mult * sslm.chain_variance)
    else:
        term1 = 0.0

    final = -(term1 + term2 + term3 + term4)

    return final


def df_obs(x, *args):
    """Derivative of the objective function which optimises obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The derivative of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    model = "DTM"
    if model == "DTM":
        deriv = sslm.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)
    elif model == "DIM":
        deriv = sslm.compute_obs_deriv_fixed(
            p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)  # noqa:F821

    return np.negative(deriv)

class DTM_para(utils.SaveLoad):
    def __init__(self,id2word=None,time_slice=None,topic_suffstats=None,num_topics=10,obs_variance=0.05,chain_variance=0.005,passes=10,random_state=None,chunksize=10):
        '''
        参数说明：
        
        第一部分：输入参数
        corpus:list，格式为(int, float)，float为词的字典代码，int为在此文档中的数量，每个文本一个list，外面再套一个list。
        time_slice:list of int，具体含义为每个值是当前时刻下的文档数量
        id2word:dict of (int,str)，int为编号，str为词
        num_topics:int，主题数量

        topic_suffstats:DTM中的ntw,K*V*T维，也就是t时刻每个词在K个主题上的计数，估计值为词总数*这个词在K个主题的估计概率
        obs_variance:用于近似真实的向前方差
        chain_variance:beta的变分分布的方差参数
        random_state:随机参数
        chunksize:文档的分块大小
        
        第二部分：self里的参数
        self.corpus:同前
        self.id2word:第一部分里的id2word
        self.vocab_len:id2word的长度，即所有词的数量
        self.corpus_len：corpus的长度，即所有时刻的文档总数
        self.time_slice:第一部分里的time_slice
        self.num_time_slices:总的时刻数
        self.num_topics:同前
        self.alphas:k维的alpha
        self.topic_chains: sslm模块运行的结果，会给出topic-word,doc-topic的概率
        self.max_doc_len:所有文档中最大的词长度
        self.sstats:同前
        self.time_doc_slice:list[time][doc]，每个时刻下，每个文档下的词数
        self.time_slice_start:[T+1维list]含义为每个时刻t起始位置的文档标号[0,30,60]
            
        下面的参数没有用到
        self.top_doc_phis = None
        self.influence = None
        self.renormalized_influence = None
        self.influence_sum_lgl = None
        '''
        #self.corpus = corpus
        
        self.id2word = id2word
        
        self.vocab_len = len(self.id2word)
        
        
        self.time_slice = time_slice
        self.num_topics = num_topics
        self.num_time_slices = len(time_slice)
        self.topic_suffstats = topic_suffstats

        time_slice_start = [sum(time_slice[0:u:1]) for u,k in enumerate(time_slice)]
        time_slice_start.append(sum(time_slice))
        self.time_slice_start = time_slice_start

        #每个主题的sslm模型的所有参数
        self.topic_chains = []
        for topic in range(num_topics):
            sslm_ = sslm(
                num_time_slices=self.num_time_slices, vocab_len=self.vocab_len, num_topics=self.num_topics,
                chain_variance=chain_variance, obs_variance=obs_variance
            )
            self.topic_chains.append(sslm_)
        #初始化e_log_prob以及obs等
            
        # fit DTM
        self.fit_lda_seq_topics(topic_suffstats)#拟合DTM参数
 
    def fit_lda_seq_topics(self, topic_suffstats):
        """Fit the sequential model topic-wise.

        Parameters
        ----------
        topic_suffstats : numpy.ndarray
            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The sum of the optimized lower bounds for all topics.

        """
        lhood = 0
        topic_suffstats = self.topic_suffstats
        for k, chain in enumerate(self.topic_chains):
            logger.info("Fitting topic number %i", k)
            lhood_term = sslm.fit_sslm(chain, topic_suffstats)
            lhood += lhood_term
        return lhood#fit_sslm输出lhood(ELBO)


    #输出函数部分
    def print_topic_times(self, topic, top_terms=20):
        """Get the most relevant words for a topic, for each timeslice. This can be used to inspect the evolution of a
        topic through time.

        Parameters
        ----------
        topic : int
            The index of the topic.
        top_terms : int, optional
            Number of most relevant words associated with the topic to be returned.

        Returns
        -------
        list of list of str
            Top `top_terms` relevant terms for the topic for each time slice.

        """
        topics = []
        for time in range(self.num_time_slices):
            topics.append(self.print_topic(topic, time, top_terms))

        return topics

    def print_topics(self, time=0, top_terms=20):
        """Get the most relevant words for every topic.

        Parameters
        ----------
        time : int, optional
            The time slice in which we are interested in (since topics evolve over time, it is expected that the most
            relevant words will also gradually change).
        top_terms : int, optional
            Number of most relevant words to be returned for each topic.

        Returns
        -------
        list of list of (str, float)
            Representation of all topics. Each of them is represented by a list of pairs of words and their assigned
            probability.

        """
        return [self.print_topic(topic, time, top_terms) for topic in range(self.num_topics)]

    def print_topic(self, topic, time=0, top_terms=20):
        """Get the list of words most relevant to the given topic.

        Parameters
        ----------
        topic : int
            The index of the topic to be inspected.
        time : int, optional
            The time slice in which we are interested in (since topics evolve over time, it is expected that the most
            relevant words will also gradually change).
        top_terms : int, optional
            Number of words associated with the topic to be returned.

        Returns
        -------
        list of (str, float)
            The representation of this topic. Each element in the list includes the word itself, along with the
            probability assigned to it by the topic.

        """
        topic = self.topic_chains[topic].e_log_prob
        topic = np.transpose(topic)
        topic = np.exp(topic[time])
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, top_terms, reverse=True)
        beststr = [(self.id2word[id_], topic[id_]) for id_ in bestn]
        return beststr

#调用方法:
from DTMpart import DTM_para #For topic-word distribution update
dtm_model_part=DTM_para(id2word=id2word,time_slice=time_slice,topic_suffstats=np.array(n_tw),num_topics= 10,obs_variance=0.05,chain_variance=1.5)
for k in range(num_topics):
    for t in range(len(time_slice)):
        topwords_tmp = dtm_model_1topic.print_topic(k, t, top_terms=V)#V是词汇表总数
        for wordtoken,prob in topwords_tmp:
            word = id2word.token2id[wordtoken]
            beta_estimated[t][word][k] = prob