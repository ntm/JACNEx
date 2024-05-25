import logging
import math
import numpy
import scipy.stats

####### JACNEx modules
import callCNVs.robustGaussianFit

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
############################################
# fitCNO:
# fit a half-normal distribution with mode=0 (ie loc=0 for scipy) to all FPMs
# in intergenicFPMs.
#
# Args:
# - intergenicFPMs numpy 2D-array of floats of size nbIntergenics * nbSamples
#   holding the FPM-normalized counts for intergenic pseudo-exons
#
# Returns (CN0scale, fpmCn0):
# - CN0scale is the scale parameter of the fitted half-normal distribution
# - fpmCn0 is the FPM threshold up to which data looks like it could very possibly
#   have been produced by the CN0 model (set to fracPPF of the inverse CDF == quantile
#   function). This will be used later for filtering NOCALL exons.
def fitCNO(intergenicFPMs):
    # fracPPF hard-coded here, should be fine and universal
    fracPPF = 0.95
    (hnormloc, hnormscale) = scipy.stats.halfnorm.fit(intergenicFPMs.ravel(), floc=0)
    fpmCn0 = scipy.stats.halfnorm.ppf(fracPPF, loc=0, scale=hnormscale)
    return (hnormscale, fpmCn0)


############################################
# fitCN2:
# for each exon (==row of FPMs), try to fit a normal distribution to the
# dominant component of the FPMs (this is our model of CN2, we assume that
# most samples are CN2), and apply  the following QC criteria, testing if:
# - exon isn't captured (median FPM <= fpmCn0)
# - fitting fails (exon is very atypical, can't make any calls)
# - CN2 model isn't supported by 50% or more of the samples
# - CN1 Gaussian (or CN2 Gaussian if isHaploid) can't be clearly distinguished
#   from CN0 model (see minZscore)
#
# Args:
# - FPMs: numpy 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is
#   the FPM-normalized count for exon e in sample s
# - clusterID: name of cluster (for logging)
# - fpmCn0, isHaploid: used for the QC criteria
#
# Return (Ecodes, CN2means, CN2sigmas):
# each is a 1D numpy.ndarray of size=nbExons, values for each exon are:
# - the status/error code:
#   E==0  if all QC criteria pass
#   E==-1 if exon isn't captured
#   E==-2 if robustGaussianFit failed
#   E==-3 if low support for CN2 model
#   E==-4 if CN1 (CN2 if isHaploid) is too close to CN0
# - the mean and stdev of the CN2 model (set to (0,0) if we couldn't fit CN2,
# ie Ecode==-1 or -2)
def fitCN2(FPMs, clusterID, fpmCn0, isHaploid):
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]

    Ecodes = numpy.zeros(nbExons, dtype=numpy.byte)
    CN2means = numpy.zeros(nbExons, dtype=numpy.float64)
    CN2sigmas = numpy.zeros(nbExons, dtype=numpy.float64)

    for ei in range(nbExons):
        if numpy.median(FPMs[ei, :]) <= fpmCn0:
            # uncaptured exon
            Ecodes[ei] = -1
            continue

        (mu, sigma) = callCNVs.robustGaussianFit.robustGaussianFit(FPMs[ei, :])
        if mu == 0:
            # cannot robustly fit Gaussian
            Ecodes[ei] = -2
            continue

        # if we get here, (mu, sigma) are OK:
        (CN2means[ei], CN2sigmas[ei]) = (mu, sigma)

        # require at least minSamps samples within sdLim sigmas of mu
        minSamps = nbSamples * 0.5
        sdLim = 2
        samplesUnderCN2 = numpy.sum(numpy.logical_and(FPMs[ei, :] - mu - sdLim * sigma < 0,
                                                      FPMs[ei, :] - mu + sdLim * sigma > 0))
        if samplesUnderCN2 < minSamps:
            # low support for CN2
            Ecodes[ei] = -3
            continue

        # require CN1 (CN2 if haploid) to be at least minZscore sigmas from fpmCn0
        minZscore = 3
        if ((not isHaploid and ((mu / 2 - minZscore * sigma) <= fpmCn0)) or
            (isHaploid and ((mu - minZscore * sigma) <= fpmCn0))):
            # CN1 / CN2 too close to CN0
            Ecodes[ei] = -4
            continue

        # if we get here exon is good, but there's nothing more to do

    return(Ecodes, CN2means, CN2sigmas)


############################################
# calcLikelihoods:
# for each exon (==row of FPMs):
# - if Ecodes[ei]==0, calculate and fill likelihoods for CN0, CN1, CN2, CN3+
# - else exon is NOCALL => set likelihoods to -1
#
# Args:
# - FPMs: 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is the FPM count
#   for exon e in sample s
# - CN0scale: as returned by fitCN0()
# - Ecodes, CN2means, CN2sigmas: as returned by fitCN2()
# - isHaploid: used in CN3+ model
#
# Returns likelihoods (allocated here):
#   numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likelihood of state cn for exon e in sample s
def calcLikelihoods(FPMs, CN0scale, Ecodes, CN2means, CN2sigmas, isHaploid):
    # sanity:
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    if nbExons != len(Ecodes):
        logger.error("sanity check failed in calcLikelihoods(), impossible!")
        raise Exception("calcLikelihoods sanity check failed")

    # allocate 3D-array (hard-coded: 4 states CN0, CN1, CN2, CN3+)
    likelihoods = numpy.full((nbSamples, nbExons, 4), fill_value=-1,
                             dtype=numpy.float64, order='C')

    # CN0 model: half-normal distribution with mode=0
    # NOTE: calculating for all exons for speed, will need to reset to -1 for NOCALL exons
    for si in range(nbSamples):
        likelihoods[si, :, 0] = scipy.stats.halfnorm.pdf(FPMs[:, si], scale=CN0scale)

    if isHaploid:
        # set all likelihoods of CN1 to zero
        likelihoods[:, :, 1] = 0

    for ei in range(nbExons):
        if Ecodes[ei] < 0:
            # exon is NOCALL for the whole cluster, squash likelihoods to -1
            likelihoods[:, ei, :] = -1.0

        else:
            # CN1: shift the CN2 Gaussian so mean==cn2Mu/2 (a single copy rather than 2)
            if not isHaploid:
                likelihoods[:, ei, 1] = gaussianPDF(FPMs[ei, :], CN2means[ei] / 2, CN2sigmas[ei])
            # else keep CN1 likelihood at zero as set above

            # CN2 model: the fitted Gaussian
            likelihoods[:, ei, 2] = gaussianPDF(FPMs[ei, :], CN2means[ei], CN2sigmas[ei])

            # CN3 model, as defined in cn3PDF()
            likelihoods[:, ei, 3] = cn3PDF(FPMs[ei, :], CN2means[ei], CN2sigmas[ei], isHaploid)

    return(likelihoods)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
############################################
# Precompute sqrt(2*pi), used tons of times in gaussianPDF()
SQRT_2PI = math.sqrt(2 * math.pi)


############################################
# Calculate the likelihoods (==values of the PDF) of a Gaussian distribution
# of parameters mu and sigma, at x (1D numpy.ndarray of floats).
# Returns a 1D numpy.ndarray, same size as x
#
# NOTE: I tried scipy.stats.norm but it's super slow, and statistics.normalDist
# but it's even slower (due to being unvectorized probably)
def gaussianPDF(x, mu, sigma):
    return numpy.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * SQRT_2PI)


############################################
# Calculate the likelihoods (==values of the PDF) of our statistical model
# of CN3+, based on the CN2 mu and sigma, at x (1D numpy.ndarray of floats).
#
# CN3+ is currently modeled as a LogNormal that aims to:
# - capture data around 1.5x the CN2 mean (2x if isHaploid) and beyond;
# - avoid overlapping too much with the CN2.
# The LogNormal is heavy-tailed, which is nice because we are modeling CN3+ not CN3.
#
# Args:
# - x: 1D numpy.ndarray of the FPMs for which we want to calculate the likelihoods
# - (cn2Mu, cn2Sigma) of the CN2 model
# - isHaploid boolean (NOTE: currently not used - TODO)
#
# Returns a 1D numpy.ndarray, same size as x
def cn3PDF(x, cn2Mu, cn2Sigma, isHaploid):
    # LogNormal parameters set empirically
    sigma = 0.5
    mu = math.log(cn2Mu)
    # shift the whole distribution by "loc" to avoid overlapping too much with CN2
    loc = cn2Mu + 2 * cn2Sigma
    # mask values <= loc to avoid doing any computations on them, this also copies the data
    xm = numpy.ma.masked_less_equal(x, loc)
    xm -= loc
    # the formula for the pdf of a LogNormal is pretty simple (see wikipedia), but
    # the order of operations is important to avoid overflows/underflows. The following
    # works for us and is reasonably fast
    res = numpy.ma.log(xm)
    res = numpy.ma.exp(-(((res - mu) / sigma)**2 / 2) - res - numpy.ma.log(sigma * SQRT_2PI))
    return (res.filled(fill_value=0.0))
