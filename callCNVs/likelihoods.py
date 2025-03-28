############################################################################################
# Copyright (C) Nicolas Thierry-Mieg and Amandine Septier, 2021-2025
#
# This file is part of JACNEx, written by Nicolas Thierry-Mieg and Amandine Septier
# (CNRS, France)  {Nicolas.Thierry-Mieg,Amandine.Septier}@univ-grenoble-alpes.fr
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
############################################################################################


import logging
import math
import numpy

####### JACNEx modules
import callCNVs.robustGaussianFit

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# fitCNO:
# fit an exponential distribution to all FPMs in intergenicFPMs.
#
# Args:
# - intergenicFPMs numpy 2D-array of floats of size nbIntergenics * nbSamples
#   holding the FPM-normalized counts for intergenic pseudo-exons
#
# Returns (CN0lambda, fpmCn0):
# - CN0lambda is the parameter of the fitted exponential distribution
# - fpmCn0 is the FPM threshold up to which data looks like it could very possibly
#   have been produced by the CN0 model (set to fracPPF of the inverse CDF == quantile
#   function). This will be used later for filtering NOCALL exons.
def fitCNO(intergenicFPMs):
    # fracPPF hard-coded here, should be fine and universal
    fracPPF = 0.99
    # maximum likelihood estimator for lambda: see wikipedia
    CN0lambda = 1.0 / intergenicFPMs.mean(dtype=numpy.float128)
    # calculate quantile function at fracPPF
    fpmCn0 = -math.log(1 - fracPPF) / CN0lambda

    return (CN0lambda, fpmCn0)


############################################
# fitCN2:
# for each exon (==row of FPMs), try to fit a Normal distribution to the
# dominant component of the FPMs (this is our model of CN2, we assume that
# most samples are CN2), and apply  the following QC criteria, testing if:
# - exon isn't captured (median FPM <= fpmCn0)
# - fitting fails (exon is very atypical, can't make any calls)
# - CN2 model isn't supported by at least minFracSamps of the samplesOfInterest
# - CN2 model can't be clearly distinguished from CN0 model (see minZscore)
# - CN1 model (in diploids only) can't be clearly distinguished from
#   CN0 or CN2 model (E==1 ie CALLED-CN1-RESTRICTED, see viterbi.py)
#
# Args:
# - FPMs: numpy 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is
#   the FPM-normalized count for exon e in sample s
# - samplesOfInterest: 1D-array of bools of size nbSamples, value==True iff the
#   sample is in the cluster of interest (vs being in a FITWITH cluster)
# - clusterID: name of cluster (for logging)
# - fpmCn0, isHaploid: used for the QC criteria
#
# Return (Ecodes, CN2mus, CN2sigmas):
# each is a 1D numpy.ndarray of size=nbExons, values for each exon are:
# - the status/error code:
#   E==-1 if exon isn't captured
#   E==-2 if fitting failed
#   E==-3 if low support for CN2 model
#   E==-4 if CN2 is too close to CN0
#   E==1  if CN1 is too close to CN0 or to CN2 but CN2 isn't (in diploids only)
#   E==0  if all QC criteria pass
# - the mu and sigma of the CN2 model, or (1,1) if we couldn't fit CN2,
#   ie Ecode==-1 or -2
#   [NOTE: (mu,sigma)==(1,1) allows to use vectorized cn2PDF() and
#   cn3PDF() without DIVBYZERO errors on NOCALL exons]
def fitCN2(FPMs, samplesOfInterest, clusterID, fpmCn0, isHaploid):
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    nbSOIs = samplesOfInterest.sum()
    # primaryCluster: True iff this cluster doesn't have FITWITHs
    primaryCluster = True
    if nbSamples != nbSOIs:
        primaryCluster = False

    Ecodes = numpy.zeros(nbExons, dtype=numpy.byte)
    CN2mus = numpy.ones(nbExons, dtype=numpy.float64)
    CN2sigmas = numpy.ones(nbExons, dtype=numpy.float64)

    # hard-coded min Z-score parameter for "too close to CN0/CN2"
    minZscore = 2
    # hard-coded fraction of samples of interest that must be "under" the CN2
    minFracSamps = 0.6

    for ei in range(nbExons):
        if numpy.median(FPMs[ei, :]) <= fpmCn0:
            # uncaptured exon
            Ecodes[ei] = -1
            continue

        # try to fit a truncated Gaussian on main component
        (mu, sigma) = callCNVs.robustGaussianFit.robustGaussianFit(FPMs[ei, :])

        if sigma == 0:
            # cannot robustly fit Gaussian
            Ecodes[ei] = -2
            continue

        # if we get here, (mu, sigma) are OK:
        (CN2mus[ei], CN2sigmas[ei]) = (mu, sigma)

        # require CN2 to be at least minZscore sigmas from fpmCn0
        if (mu - minZscore * sigma) <= fpmCn0:
            # CN2 too close to CN0
            Ecodes[ei] = -4
            continue

        # require at least minFracSamps samples (of interest) within sdLim sigmas of mu
        minSamps = int(nbSOIs * minFracSamps)
        sdLim = 2
        if primaryCluster:
            samplesUnderCN2 = numpy.sum(numpy.abs(FPMs[ei, :] - mu) < sdLim * sigma)
        else:
            samplesUnderCN2 = numpy.sum(numpy.abs(FPMs[ei, samplesOfInterest] - mu) < sdLim * sigma)

        if (samplesUnderCN2 < minSamps):
            # low support for CN2
            Ecodes[ei] = -3
            continue

        # in diploids: prefer if CN1 is at least minZscore sigmas_cn1 from fpmCn0,
        # ie mu/2 - minZscore*sigma/2 > fpmCn0, ie mu - minZscore*sigma > 2*fpmCn0
        if (not isHaploid) and ((mu - minZscore * sigma) <= 2 * fpmCn0):
            # CN1 too close to CN0
            Ecodes[ei] = 1
            continue
        # in diploids, also prefer if CN1 is at least minZscore sigmas_cn2 from CN2, ie
        # minZscore*sigma < mu/2, ie 2*minZscore*sigma < mu
        if (not isHaploid) and ((2 * minZscore * sigma) >= mu):
            # CN1 too close to CN2
            Ecodes[ei] = 1
            continue

        # if we get here exon is good, but there's nothing more to do

    return(Ecodes, CN2mus, CN2sigmas)


############################################
# calcLikelihoods:
# for each exon (==row of FPMs):
# - if Ecodes[ei] >= 0, calculate and fill likelihoods for CN0, CN1, CN2, CN3+
# - else exon is NOCALL => set likelihoods to -1 (except if forPlots)
# NOTE: if isHapoid, likelihoods[CN1]=0.
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is the FPM count
#   for exon e in sample s
# - CN0lambda: as returned by fitCN0()
# - Ecodes, CN2mus, CN2sigmas: as returned by fitCN2()
# - isHaploid: boolean (used in the CN3+ model and for zeroing CN1 likelihoods)
# - forPlots: boolean if True don't set likelihoods to -1 for NOCALL exons
#
# Returns likelihoods (allocated here):
#   numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likelihood of state cn for exon e in sample s
def calcLikelihoods(FPMs, CN0lambda, Ecodes, CN2mus, CN2sigmas, fpmCn0, isHaploid, forPlots):
    # sanity:
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    if nbExons != len(Ecodes):
        logger.error("sanity check failed in calcLikelihoods(), impossible!")
        raise Exception("calcLikelihoods sanity check failed")

    # allocate 3D-array (hard-coded: 4 states CN0, CN1, CN2, CN3+)
    likelihoods = numpy.full((nbSamples, nbExons, 4), fill_value=-1,
                             dtype=numpy.float64, order='C')

    # NOTE: calculating likelihoods for all exons, taking advantage of numpy vectorization;
    # afterwards we set them to -1 for NOCALL exons

    likelihoods[:, :, 0] = cn0PDF(FPMs, CN0lambda)

    if isHaploid:
        # in haploids: all-zeroes for CN1
        likelihoods[:, :, 1] = 0
    else:
        likelihoods[:, :, 1] = cn1PDF(FPMs, CN2mus, CN2sigmas)

    likelihoods[:, :, 2] = cn2PDF(FPMs, CN2mus, CN2sigmas, fpmCn0)
    likelihoods[:, :, 3] = cn3PDF(FPMs, CN2mus, CN2sigmas, fpmCn0, isHaploid)

    if not forPlots:
        # NOCALL exons: set to -1 for every sample+state
        likelihoods[:, Ecodes < 0, :] = -1.0

    return(likelihoods)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# Precompute sqrt(2*pi), used many times
SQRT_2PI = math.sqrt(2 * math.pi)


############################################
# Calculate the likelihoods (==values of the PDF) of a Gaussian distribution
# of parameters mu[e] and sigma[e], at FPMs[e,:] for every exonIndex e.
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - mu, sigma: 1D-arrays of floats of size nbExons
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs gets transposed)
#
# NOTE: I tried scipy.stats.norm but it's super slow, and statistics.normalDist
# but it's even slower (due to being unvectorized probably). This code is fast.
def gaussianPDF(FPMs, mu, sigma):
    # return numpy.exp(-0.5 * ((FPMs.T - mu) / sigma)**2) / (sigma * SQRT_2PI)
    res = FPMs.T - mu
    res /= sigma
    res *= res
    res /= -2
    res = numpy.exp(res)
    res /= (sigma * SQRT_2PI)
    return(res)


############################################
# Calculate the likelihoods (==values of the PDF) of a LogNormal distribution
# of parameters mu[e] and sigma[e] and starting at location "shift", at FPMs[e,:]
# for every exonIndex e.
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - mu, sigma: 1D-arrays of floats of size nbExons
# - shift: float
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs gets transposed)
def logNormalPDF(FPMs, mu, sigma, shift):
    # mask values <= shift to avoid doing any computations on them, this also copies the data
    res = numpy.ma.masked_less_equal(FPMs.T, shift)
    res -= shift
    # the formula for the pdf of a LogNormal is pretty simple (see wikipedia), but
    # the order of operations is important to avoid overflows/underflows. The following
    # works for us and is reasonably fast
    res = numpy.ma.log(res)
    res = numpy.ma.exp(-(((res - mu) / sigma)**2 / 2) - res - numpy.ma.log(sigma * SQRT_2PI))
    return (res.filled(fill_value=0.0))


############################################
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN0 at every datapoint in FPMs.
# Our current CN0 model is an exponential distribution of parameter CN0lambda
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - CN0lambda: parameter of the CN0
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs get transposed)
def cn0PDF(FPMs, CN0lambda):
    # pdf(X) = lambda * numpy.exp(-lambda * X)
    res = numpy.exp(-CN0lambda * FPMs.T)
    res *= CN0lambda
    return (res)


############################################
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN1, based on the CN2 parameters, at FPMs[e,:] for every exonIndex e.
# Our current CN1 model is a Normal distribution of parameters
# (mu_CN2 / 2, sigma_CN2 / 2) ie model a single copy rather than 2
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - cn2Mu, cn2Sigma: 1D-arrays of floats of size nbExons
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs get transposed)
def cn1PDF(FPMs, cn2Mu, cn2Sigma):
    return(gaussianPDF(FPMs, cn2Mu / 2, cn2Sigma / 2))


############################################
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN2 at FPMs[e,:] for every exonIndex e.
# Our current CN2 model is a Normal distribution of parameters
# (mu_CN2, sigma_CN2), except it is "squashed" for small FPMs (to avoid masking
# some genuine single-exon HOMODELs in some specific conditions)
# NOTE: because of the squashing the function isn't strictly a PDF (it doesn't
# integrate to 1), but this is marginal and not worth renormalizing.
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - cn2Mu, cn2Sigma: 1D-arrays of floats of size nbExons
# - fpmCn0: float
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs get transposed)
def cn2PDF(FPMs, cn2Mu, cn2Sigma, fpmCn0):
    # for fpm < fpmCn0 we want to squash the likelihood, and we want a continuous
    # function around fpmCn0 -> using a power-law attenuation:
    # for x >= fpmCn0 : f(x) = Norm(x)
    # for x < fpmCn0 : f(x) = Norm(x) * [Norm(x) / Norm(fpmCn0)]**attenuationPower
    attenuationPower = 3
    res = gaussianPDF(FPMs, cn2Mu, cn2Sigma)
    normFpmCn0 = gaussianPDF(numpy.array([fpmCn0]), cn2Mu, cn2Sigma)
    # special case: if normFpmCn0==0 we would divideByZero below, but in this case
    # resSmallFpms for this exon is all-zeroes anyway -> set normFpmCn0=1
    normFpmCn0[normFpmCn0 == 0] = 1

    # copy=False, we will modify res in-place
    resSmallFpms = numpy.ma.masked_where(FPMs.T >= fpmCn0, res, copy=False)
    normFact = resSmallFpms / normFpmCn0
    normFact **= attenuationPower
    resSmallFpms *= normFact

    return(res)


############################################
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN3+, based on the CN2 mu's and sigmas, at FPMs[e,:] for every exonIndex e.
#
# CN3+ is currently modeled as a LogNormal that aims to:
# - capture data starting around 1.5x the CN2 mean (2x if isHaploid);
# - avoid overlapping too much with the CN2.
# The LogNormal is heavy-tailed, which is nice because we are modeling CN3+ not CN3.
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - cn2Mu, cn2Sigma: 1D-arrays of floats of size nbExons
# - isHaploid boolean
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs gets transposed)
def cn3PDF(FPMs, cn2Mu, cn2Sigma, fpmCn0, isHaploid):
    # shift the whole distribution to avoid overlapping too much with CN2
    shift = cn2Mu
    # LogNormal parameters set empirically
    sigma = cn2Sigma
    mu = numpy.log(cn2Mu) + sigma * sigma
    # These (sigma,mu) result in a logNormal whose mode is cn2Mu + shift == 2*cn2Mu ,
    # this should be good for diploids (mode at CN==4), but for haploids this would
    # place the mode at CN==2 ie the lowest possible number of copies for a DUP,
    # we prefer a mode at CN==3 ie mu += ln(2)
    if isHaploid:
        mu += math.log(2)

    return(logNormalPDF(FPMs, mu, sigma, shift))
