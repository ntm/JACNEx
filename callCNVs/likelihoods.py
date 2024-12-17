############################################################################################
# Copyright (C) Nicolas Thierry-Mieg and Amandine Septier, 2021-2024
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
import pyerf

####### JACNEx modules
import callCNVs.robustGaussianFit

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# fitCNO:
# fit a half-normal distribution to all FPMs in intergenicFPMs.
#
# Args:
# - intergenicFPMs numpy 2D-array of floats of size nbIntergenics * nbSamples
#   holding the FPM-normalized counts for intergenic pseudo-exons
#
# Returns (CN0sigma, fpmCn0):
# - CN0sigma is the parameter of the fitted half-normal distribution
# - fpmCn0 is the FPM threshold up to which data looks like it could very possibly
#   have been produced by the CN0 model (set to fracPPF of the inverse CDF == quantile
#   function). This will be used later for filtering NOCALL exons.
def fitCNO(intergenicFPMs):
    # fracPPF hard-coded here, should be fine and universal
    fracPPF = 0.95
    # bias-corrected maximum likelihood estimator for sigma: see wikipedia
    # calculate MLE estimate of sigma
    N = intergenicFPMs.ravel().shape[0]
    sigma = math.sqrt((intergenicFPMs.ravel() * intergenicFPMs.ravel()).sum() / N)
    # correct for bias
    sigma += sigma / 4 / N

    # calculate quantile function at fracPPF
    fpmCn0 = sigma * math.sqrt(2) * pyerf.erfinv(fracPPF)

    return (sigma, fpmCn0)


############################################
# fitCN2:
# for each exon (==row of FPMs), try to fit a normal distribution to the
# dominant component of the FPMs (this is our model of CN2, we assume that
# most samples are CN2), and apply  the following QC criteria, testing if:
# - exon isn't captured (median FPM <= fpmCn0)
# - fitting fails (exon is very atypical, can't make any calls)
# - CN2 model isn't supported by at least minFracSamps of the samplesOfInterest
# - CN2 Gaussian can't be clearly distinguished from CN0 model (see minZscore)
# - CN1 Gaussian (in diploids only) can't be clearly distinguished from
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
# Return (Ecodes, CN2means, CN2sigmas):
# each is a 1D numpy.ndarray of size=nbExons, values for each exon are:
# - the status/error code:
#   E==-1 if exon isn't captured
#   E==-2 if robustGaussianFit failed
#   E==-3 if low support for CN2 model
#   E==-4 if CN2 is too close to CN0
#   E==1  if CN1 is too close to CN0 or to CN2 but CN2 isn't (in diploids only)
#   E==0  if all QC criteria pass
# - the mean and stdev of the CN2 model, or (1,1) if we couldn't fit CN2,
#   ie Ecode==-1 or -2
#   [NOTE: (mean,stdev)==(1,1) allows to use vectorized gaussianPDF() and
#   cn3PDF() without DIVBYZERO errors on NOCALL exons]
def fitCN2(FPMs, samplesOfInterest, clusterID, fpmCn0, isHaploid):
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]

    Ecodes = numpy.zeros(nbExons, dtype=numpy.byte)
    CN2means = numpy.ones(nbExons, dtype=numpy.float64)
    CN2sigmas = numpy.ones(nbExons, dtype=numpy.float64)

    # hard-coded min Z-score parameter for "too close to CN0/CN2"
    minZscore = 2
    # hard-coded fraction of samples of interest that must be "under" the CN2
    minFracSamps = 0.5

    for ei in range(nbExons):
        if numpy.median(FPMs[ei, :]) <= fpmCn0:
            # uncaptured exon
            Ecodes[ei] = -1
        else:
            (mu, sigma) = callCNVs.robustGaussianFit.robustGaussianFit(FPMs[ei, :])
            if mu == 0:
                # cannot robustly fit Gaussian
                Ecodes[ei] = -2
            else:
                # if we get here, (mu, sigma) are OK:
                (CN2means[ei], CN2sigmas[ei]) = (mu, sigma)

                # require at least minFracSamps samples of interest within sdLim sigmas of mu
                minSamps = nbSamples * minFracSamps
                sdLim = 2
                samplesUnderCN2 = numpy.sum(numpy.logical_and(FPMs[ei, samplesOfInterest] - mu - sdLim * sigma < 0,
                                                              FPMs[ei, samplesOfInterest] - mu + sdLim * sigma > 0))
                if samplesUnderCN2 < minSamps:
                    # low support for CN2
                    Ecodes[ei] = -3

                # require CN2 to be at least minZscore sigmas from fpmCn0
                elif (mu - minZscore * sigma) <= fpmCn0:
                    # CN2 too close to CN0
                    Ecodes[ei] = -4

                # in diploids: prefer if CN1 is at least minZscore sigmas_cn1 from fpmCn0,
                # ie mu/2 - minZscore*sigma/2 > fpmCn0, ie mu - minZscore*sigma > 2*fpmCn0
                elif (not isHaploid) and ((mu - minZscore * sigma) <= 2 * fpmCn0):
                    # CN1 too close to CN0
                    Ecodes[ei] = 1
                # in diploids, also prefer if CN1 is at least minZscore sigmas_cn2 from CN2, ie
                # minZscore*sigma < mu/2, ie 2*minZscore*sigma < mu
                elif (not isHaploid) and ((2 * minZscore * sigma) >= mu):
                    # CN1 too close to CN2
                    Ecodes[ei] = 1

                else:
                    # if we get here exon is good, but there's nothing more to do
                    pass

    return(Ecodes, CN2means, CN2sigmas)


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
# - CN0sigma: as returned by fitCN0()
# - Ecodes, CN2means, CN2sigmas: as returned by fitCN2()
# - isHaploid: boolean (used in the CN3+ model and for zeroing CN1 likelihoods)
# - forPlots: boolean if True don't set likelihoods to -1 for NOCALL exons
#
# Returns likelihoods (allocated here):
#   numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likelihood of state cn for exon e in sample s
def calcLikelihoods(FPMs, CN0sigma, Ecodes, CN2means, CN2sigmas, isHaploid, forPlots):
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

    # CN0 model, as defined in cn0PDF()
    likelihoods[:, :, 0] = cn0PDF(FPMs, CN0sigma)

    # CN1:
    if isHaploid:
        # in haploids: all-zeroes
        likelihoods[:, :, 1] = 0
    else:
        # in diploids: rescale the CN2 Gaussian by factor 1/2 (model a single copy rather
        # than 2) -> Normal(CN2mu/2, CN2sigma/2)
        likelihoods[:, :, 1] = gaussianPDF(FPMs, CN2means / 2, CN2sigmas / 2)

    # CN2 model: the fitted CN2 Gaussian
    likelihoods[:, :, 2] = gaussianPDF(FPMs, CN2means, CN2sigmas)

    # CN3 model, as defined in cn3PDF()
    likelihoods[:, :, 3] = cn3PDF(FPMs, CN2means, CN2sigmas, isHaploid)

    if not forPlots:
        # NOCALL exons: set to -1 for every sample+state
        likelihoods[:, Ecodes < 0, :] = -1.0

    return(likelihoods)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# Precompute sqrt(2*pi), used many times in gaussianPDF() and cn3PDF()
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
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN0 at every datapoint in FPMs.
# Our current CN0 model is a half-normal distribution of parameter sigma,
# truncated at 4*sigma (corresponds to CDF > 0.99993)
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples
# - sigma: parameter of the CN0
#
# Returns a 2D numpy.ndarray of size nbSamples * nbExons (FPMs gets transposed)
def cn0PDF(FPMs, sigma):
    # numpy.exp(-0.5 * (X / sigma)**2) * (2 / sigma / SQRT_2PI)
    res = FPMs.T / sigma
    res *= res
    res /= -2
    res = numpy.exp(res)
    res *= 2 / sigma / SQRT_2PI
    # truncate at 4*sigma
    res[FPMs.T >= 4 * sigma] = 0.0
    return (res)


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
def cn3PDF(FPMs, cn2Mu, cn2Sigma, isHaploid):
    # shift the whole distribution by -loc to avoid overlapping too much with CN2
    loc = cn2Mu + 2 * cn2Sigma
    # LogNormal parameters set empirically
    sigma = 0.5
    mu = numpy.log(cn2Mu)
    # These (sigma,mu) result in ~8.3% of the distrib below 3*cn2Mu/2 + 2*cn2Sigma,
    # i.e. we don't capture too much stuff before what really looks like CN=3
    if isHaploid:
        # diploid aims for 3*cn2Mu/2 while haploid aims for 2*cn2Mu, and the data is
        # shifted by loc ~= cn2Mu (ignoring the 2 cn2Sigmas here) => rescale the
        # distribution by a factor of 2 <=> mu += ln(2)
        mu += math.log(2)

    # mask values <= loc to avoid doing any computations on them, this also copies the data
    res = numpy.ma.masked_less_equal(FPMs.T, loc)
    res -= loc
    # the formula for the pdf of a LogNormal is pretty simple (see wikipedia), but
    # the order of operations is important to avoid overflows/underflows. The following
    # works for us and is reasonably fast
    res = numpy.ma.log(res)
    res = numpy.ma.exp(-(((res - mu) / sigma)**2 / 2) - res - numpy.ma.log(sigma * SQRT_2PI))
    return (res.filled(fill_value=0.0))
