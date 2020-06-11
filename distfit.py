#!/usr/bin/python3

"""
Distribution Fit Program

Bruce Rushing

6/10/2020

Estimates a generating distribution from a data sample.

Meant to be run in python3 or later.
"""


import argparse
import numpy as np
import math
from numpy import linalg as la
from scipy import stats

# True parameters
ALPHA = 0.0
BETA = 4.0
MU = 1.8
SIGMA = 1.5
LAMBDA = 1.3

# Constants
SMALL_N = 0.00000000000000000000000000000000000000000001
SIZE = 10000
ETA = 0.01
EPSILON = 0.00000001
MAX_ITER = 1000

# MLE Functions
def build_sample_uniform(length):
    return np.random.uniform(ALPHA, BETA, [length])

def build_sample_normal(length):
    return np.random.normal(MU,SIGMA,[length])

def build_sample_exponential(length):
    return np.random.exponential((1 / LAMBDA), [length])

def MLE_uniform(data):
    alpha = np.min(data)
    beta = np.max(data)
    return np.asarray([alpha, beta])

def MLE_normal(data):
    """
    #Code for running with gradient descent.
    x_0 = np.asarray([50.0, 50.0])
    l_nabla = lambda theta: loglikelihood_normal_fpp(data, theta)
    mle = gradient_ascent_func(x_0, l_nabla)
    return mle
    """
    mu = np.mean(data)
    sigma = np.sqrt(np.var(data))
    return np.asarray([mu, sigma])

def MLE_exp(data):
    """
    #Code for running based on analytic proof.
    lamb = 1 / np.mean(data)
    return np.asarray([lamb])
    """
    x_0 = np.asarray([0.01])
    l_p = lambda theta: loglikelihood_exp_fp(data, theta)
    l_pp = lambda theta: loglikelihood_exp_fpp(data, theta)

    mle = newtons_method_func(x_0, l_p, l_pp)
    return mle

def likelihood_uniform(x_bar, theta_bar):
    def pdf(x):
        if x >= alpha and x <= beta:
            return 1 / (beta - alpha)
        else:
            return 0

    alpha = theta_bar[0]
    beta = theta_bar[1]

    pdf_v = np.vectorize(pdf)
    return np.product(pdf_v(x_bar))

def likelihood_normal(x_bar, theta_bar):
    mu = theta_bar[0]
    sigma = theta_bar[1]

    pdf = lambda x: stats.norm.pdf(x, loc=mu, scale=sigma)
    pdf_v = np.vectorize(pdf)
    return np.product(pdf_v(x_bar))

def likelihood_exp(x_bar, theta_bar):
    lamb = 1 / theta_bar[0]
    min = np.min(x_bar)

    pdf = lambda x: stats.expon.pdf(x, loc=min, scale=lamb)
    pdf_v = np.vectorize(pdf)
    return np.product(pdf_v(x_bar))

def loglikelihood_uniform_func(x_bar, theta_bar):
    alpha = theta_bar[0]
    beta = theta_bar[1]

    def f(x):
        if x >= alpha and x <= beta:
            return np.log(1.0 / (beta - alpha))
        else:
            return np.log(SMALL_N)

    fv = np.vectorize(f)
    likelihood = np.sum(fv(x_bar))
    return likelihood

def loglikelihood_uniform_fpp(x_bar, theta_bar):
    alpha = theta_bar[0]
    beta = theta_bar[1]

    f = lambda x: 1 / ((beta - alpha) ** 2)
    fv = np.vectorize(f)
    likelihood = np.sum(fv(x_bar))
    return np.asarray([likelihood, likelihood])

def loglikelihood_normal_func(x_bar, theta_bar):
    mu = theta_bar[0]
    sigma = theta_bar[1]
    n = np.size(x_bar)

    f = lambda x: (x - mu) ** 2
    fv = np.vectorize(f)
    likelihood = ((-n / 2) * np.log(2 * np.pi * sigma)) - ((1 / (2 * sigma)) *
            np.sum(fv(x_bar)))
    return likelihood

def loglikelihood_normal_fp(x_bar, theta_bar):
    mu = theta_bar[0]
    sigma = theta_bar[1]
    n = np.size(x_bar)

    f = lambda x: x - mu
    g = lambda x: (x - mu) ** 2
    fv = np.vectorize(f)
    gv = np.vectorize(g)
    likelihood_mu = (1 / sigma) * np.sum(fv(x_bar))
    likelihood_sigma = (-n / (2 * sigma)) + ((1 / (2 * (sigma ** 2))) *
            (np.sum(gv(x_bar))))
    return np.asarray([likelihood_mu, likelihood_sigma])

def loglikelihood_normal_fpp(x_bar, theta_bar):
    mu = theta_bar[0]
    sigma = theta_bar[1]

    n = np.size(x_bar)
    f = lambda x: (x - mu) ** 2
    fv = np.vectorize(f)
    likelihood_mu = -n / (sigma ** 2)
    likelihood_sig = (n / (sigma ** 2)) + ((1 / (sigma ** 3)) * np.sum(fv(x_bar)))
    return np.asarray([likelihood_mu, likelihood_sig])

def loglikelihood_exp_func(x_bar, theta_bar):
    lamb = theta_bar[0]

    def f(x):
        if x >= 0.0:
            return np.log(lamb) - (lamb * x)
        else:
            return np.log(SMALL_N)

    fv = np.vectorize(f)
    likelihood = np.sum(fv(x_bar))
    return likelihood

def loglikelihood_exp_fp(x_bar, theta_bar):
    lamb = theta_bar[0]
    n = np.size(x_bar)
    sum_x = np.sum(x_bar)

    f = (n / lamb) - sum_x
    return np.asarray([f])

def loglikelihood_exp_fpp(x_bar, theta_bar):
    lamb = theta_bar[0]
    n = np.size(x_bar)
    f = ((-1 * n) / (lamb ** 2))
    return np.asarray([f])

def newtons_method_func(x_0, f, fp):

    def exit_func(steps, y):
        """
        Tells the algorithm to stop when percentage change between each iteration
        crosses below some epsilon.
        """

        y_p = y - (f(y) / fp(y))

        # Check to see if y_p = 0
        if y_p != 0.0:
            distance = la.norm((y_p - y)) / la.norm(y_p)
        else:
            distance = 0.0

        # Check the exit condition
        if distance < EPSILON:
            return False
        elif steps > MAX_ITER:
            return False
        else:
            return True

    x = np.asarray(x_0, dtype=np.double)
    count = 0

    while exit_func(count, x):
        count += 1
        # Check to see if fp == 0
        if np.all(fp(x) == 0.0) or x == 0.0:
            break

        # Compute next iteration
        #print('First derivative = ' + str(f(x)))
        #print('Second derivative = ' + str(fp(x)))
        x = x - (f(x) / fp(x))
        if np.all(f(x) == 0.0):
            break

    return x

def gradient_ascent_func(x_0, nabla_f):

    def exit_func(steps, y):
        """
        Tells the algorithm to stop when percentage change between each iteration
        crosses below some epsilon.
        """
        y_p = y + (ETA * nabla_f(y))

        # Check to see if y_p = 0
        if la.norm(y_p) != 0.0:
            distance = la.norm((y_p - y)) / la.norm(y_p)
        else:
            distance = 0.0

        # Check the exit condition
        if distance < EPSILON:
            return False
        elif steps > MAX_ITER:
            return False
        else:
            return True

    x = np.asarray(x_0, dtype=np.double)
    count = 0

    while exit_func(count, x):
        count += 1
        # Compute next iteration
        x = x + (ETA * nabla_f(x))

    return x

def confidence_interval_func(estimate, c, ldp):
    alpha = (1 - c) / 2
    z = np.abs(stats.norm.ppf(alpha))

    lower = estimate - (z * (1 / np.sqrt(np.abs(ldp(estimate)))))
    upper = estimate + (z * (1 / np.sqrt(np.abs(ldp(estimate)))))

    return [lower, upper]

def pvalue_func(sample, mean, null_cdf):
    z_score = (np.mean(sample) - mean) / stats.sem(sample)

    if z_score > 0:
        p = 1 - null_cdf(z_score) + null_cdf(-1 * z_score)
    else:
        p = 1 + null_cdf(z_score) - null_cdf(-1 * z_score)

    return p

def pvalue_ks_func(sample, null_cdf):
    ks, p_ks = stats.kstest(sample, null_cdf)
    return p_ks

def bayes_thm_func(likelihoods, prior, estimate=0):
    """
    Calculates a posterior given likelihood pdfs and a uniform prior on those pdfs.

    Calculations performed with logarithmic likelihoods and probabilities.
    Returned value is raised by e to return probabilities.
    """
    # Calculate evidence
    log_prior = np.log(prior, dtype=np.longdouble)
    p_numerator = np.longdouble(likelihoods[estimate] + log_prior)
    p_remainder = np.longdouble(1.0)
    p_evidence = p_numerator
    for i in range(np.size(likelihoods)):
        if i != estimate:
            p_exponent = np.longdouble(likelihoods[i]) + log_prior - p_numerator
            # If-statement to catch overflow error for what is representable using longdouble
            if p_exponent < 11356:
                p_remainder += np.exp(p_exponent, dtype=np.longdouble)
            else:
                p_remainder += np.exp(11354, dtype=np.longdouble)

    p_evidence += np.log(p_remainder)
    posterior = p_numerator - p_evidence
    return np.exp(posterior)

def bayes_factor_func(likelihoods, estimate=0):
    """
    Returns Bayes factors in logarithmic form.
    """
    factors = np.asarray([])
    for like in likelihoods:
        if like != likelihoods[estimate]:
            if like == 0.0:
                bayes_factor = float('inf')
            else:
                bayes_factor = likelihoods[estimate] - like
            factors = np.append(factors, bayes_factor)

    return factors

def compute_stats_func(sample, estimates, c):
    """
    Computes a series of statistics and outputs a table with those statistics.
    Returns a table.

    The statistics that are computed and displayed are:
    1. Likelihoods
    2. Z-Test P-values
    3. K-S P-values
    4. Posterior on Uniform Distribution
    5. Bayes Factor on Maximal Likelihood
    6. Confidence Interval on c for maximum likelihood parameters

    Inputs are the data sample, a dictionary of estimates, a confidence level
    (float), and a dictionary of second derivatives of likelihood function.

    The structure of the dictionary estimates has to be the following:
    key = distribution type, values = list with parameters, likelihood,
    loglikelihood, cdf, ldp list.
    """
    def key(x):
        return x[1][2]

    # Sort the distributions on the loglikelihood
    ranked_estimates = dict(sorted(estimates.items(), key=key, reverse=True))

    # Collect likelihoods
    prior = stats.uniform.pdf(1, loc=0, scale=3)
    likelihoods = np.asarray([])
    for e in ranked_estimates:
        likelihoods = np.append(likelihoods, ranked_estimates[e][2])

    table = '{:<50}'.format('Distribution') + '{:<30}'.format('Log-Likelihood') + \
        '{:<30}'.format('Z-test P-value') + '{:<30}'.format('KS P-value') + \
        '{:<30}'.format('Posterior') + 'Average Bayes Factor\n'
    tally = -100000000000000000.0
    for e in ranked_estimates:
        # Notify the maximum item in the dictionary
        if ranked_estimates[e][2] > tally:
            tally = ranked_estimates[e][2]
            marker = e

        # Name the estimate
        table += '{:<50}'.format(e + ' ' + str(ranked_estimates[e][0]))

        # Add the likelihood
        table += '{:<30}'.format(str(ranked_estimates[e][2]))

        # Compute and add the p-value
        if e == 'Uniform':
            mean = (ranked_estimates[e][0][1] + ranked_estimates[e][0][0]) / 2
        elif e == 'Exponential':
            mean = np.min(sample) + (1 / ranked_estimates[e][0][0])
        else:
            mean = ranked_estimates[e][0][0]

        p_value = pvalue_func(sample, mean, ranked_estimates[e][3])
        table += '{:<30}'.format(str(p_value))

        # Compute KS p-value
        ks_p_value = pvalue_ks_func(sample, ranked_estimates[e][3])
        table += '{:<30}'.format(str(ks_p_value))

        # Posterior
        estimate = 0
        for i in range(np.size(likelihoods)):
            if ranked_estimates[e][2] == likelihoods[i]:
                estimate = i
        posterior = bayes_thm_func(likelihoods, prior, estimate=estimate)
        table += '{:<30}'.format(str(posterior))

        # Bayes Factor
        bayes_factor = bayes_factor_func(likelihoods, estimate=estimate)
        table += str(bayes_factor) + '\n'

    # Confidence Interval on c
    for i in range(np.size(ranked_estimates[marker][0])):
        par = ranked_estimates[marker][0][i]
        ldp = ranked_estimates[marker][4][i]
        c_interval = confidence_interval_func(par, c, ldp)
        table += 'Confidence Interval for the parameters is on ' +str(c) + \
                ' is ' + str(c_interval) + '\n'

    return table

def import_data_func(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [x.strip() for x in lines]
    sample = [float(x) for x in lines]
    return np.asarray(sample)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fits a distribution to a file of floats.')
    parser.add_argument('confidence', choices=[0.9, 0.95, 0.99, 0.999], type=float, default=0.95,
                        help='A confidence level for outputting confidence intervals')
    parser.add_argument('filename', type=str, help='A file name of floats.')

    args = parser.parse_args()

    filename = args.filename
    c = args.confidence
    sample1 = import_data_func(filename)

    sample_list = [sample1]

    for s in sample_list:
        guess1 = MLE_uniform(s)
        guess2 = MLE_normal(s)
        guess3 = MLE_exp(s)

        # Calculate likelihoods and loglikelihoods
        like_unif = likelihood_uniform(s, guess1)
        like_norm = likelihood_normal(s, guess2)
        like_exp = likelihood_exp(s, guess3)
        loglike_unif = loglikelihood_uniform_func(s, guess1)
        loglike_norm = loglikelihood_normal_func(s, guess2)
        loglike_exp = loglikelihood_exp_func(s, guess3)

        # Build CDFs for likelihood distributions
        uniform_cdf = lambda x: stats.uniform.cdf(x, loc=guess1[0], scale=guess1[1])
        normal_cdf = lambda x: stats.norm.cdf(x, loc=guess2[0], scale=guess2[1])
        exp_cdf = lambda x: stats.expon.cdf(x, loc=np.min(s), scale=(1 / guess3[0]))

        # Build likelihood second partial deriviates for confidence intervals
        ldp_a = lambda a: loglikelihood_uniform_fpp(s, np.asarray([a, guess1[1]]))[0]
        ldp_b = lambda b: loglikelihood_uniform_fpp(s, np.asarray([guess1[0], b]))[1]
        ldp_mu = lambda mu: loglikelihood_normal_fpp(s, np.asarray([mu, guess2[1]]))[0]
        ldp_sig = lambda sig: loglikelihood_normal_fpp(s, np.asarray([guess2[0], sig]))[1]
        ldp_lam = lambda l: loglikelihood_exp_fpp(s, np.asarray([l]))[0]

        # Build dictionaries
        estimate_dict = {'Uniform': [guess1, like_unif, loglike_unif, uniform_cdf, [ldp_a, ldp_b]], \
                        'Normal': [guess2, like_norm, loglike_norm, normal_cdf, [ldp_mu, ldp_sig]], \
                        'Exponential': [guess3, like_exp, loglike_exp, exp_cdf, [ldp_lam]]}

        # Compute statistics and table
        table = compute_stats_func(s, estimate_dict, c)
        print(table)
