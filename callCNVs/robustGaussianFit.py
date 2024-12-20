# Content of this file adapted from https://github.com/hmiemad/robust_Gaussian_fit ,
# thanks to the author for sharing.
#
# Copyright (c) 2022 hmiemad
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import math
import numba
import numpy

# silence numba when we are in DEBUG mode
logging.getLogger('numba').setLevel(logging.INFO)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################################################
def normal_erf(x, mu=0.0, sigma=1.0, depth=50):
    ele = 1.0
    normal = 1.0
    x = (x - mu) / sigma
    erf = x
    for i in range(1, depth):
        ele *= -x**2 / 2.0 / i
        normal += ele
        erf += ele * x / (2.0 * i + 1)

    return (max(normal / math.sqrt(2 * math.pi) / sigma, 0),
            min(max(erf / math.sqrt(2 * math.pi) / sigma, -0.5), 0.5))


#############################################################
def truncated_integral_and_sigma(x):
    n, e = normal_erf(x)
    return math.sqrt(1 - n * x / e)


#############################################################
# set bandwidth module-wide
BANDWIDTH = 2.5
# Precompute truncated_integral_and_sigma with the hard-coded bandwidth
TRUNCINTSIG = truncated_integral_and_sigma(BANDWIDTH)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#############################################################
@numba.njit
def robustGaussianFit(X, mu=None, sigma=None, bandwidth=BANDWIDTH, eps=1.0e-5):
    """
    Fits a single principal gaussian component around a starting guess point
    in a 1-dimensional gaussian mixture of unknown components with EM algorithm

    Args:
        X (numpy.array): A sample of 1-dimensional mixture of gaussian random variables
        mu (float, optional): Expectation. Defaults to None.
        sigma (float, optional): Standard deviation. Defaults to None.
        bandwidth (float, optional): Hyperparameter of truncation. Defaults to BANDWIDTH.
        eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.

    Returns:
        mu,sigma: mean and stdev of the gaussian component, or (0,0) if something failed
    """

    if mu is None:
        # median is an approach as robust and naïve as possible to Expectation
        mu = numpy.median(X)
        if mu == 0:
            return(0, 0)
    mu_0 = mu + 1

    if sigma is None:
        # rule of thumb
        sigma = numpy.std(X) / 3
    sigma_0 = sigma + 1

    # we will use TRUNCINTSIG -> avoid calculating the same thing tons of times
    # bandwidth_truncated_normal_sigma = truncated_integral_and_sigma(bandwidth)

    while abs(mu - mu_0) + abs(sigma - sigma_0) > eps:
        # loop until tolerence is reached
        """
        create a uniform window on X around mu of width 2*bandwidth*sigma
        find the mean of that window to shift the window to most expected local value
        measure the standard deviation of the window and divide by the stddev of a truncated gaussian distribution
        """
        window = numpy.logical_and(X - mu - bandwidth * sigma < 0, X - mu + bandwidth * sigma > 0)
        if not window.any():
            return(0, 0)
        mu_0 = mu
        mu = numpy.average(X[window])
        sigma_0 = sigma
        sigma = numpy.std(X[window]) / TRUNCINTSIG

    if sigma == 0:
        # set to 5% on each side of the mean
        sigma = mu * 0.05
    return (mu, sigma)
