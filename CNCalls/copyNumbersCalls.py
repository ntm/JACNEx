import sys
import os
import logging
import numpy as np
import scipy.stats as st
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import time

import clusterSamps.smoothing

# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#####################################
# allocateLogOddsArray:
# Args:
# - exons (dict[str, List[int]]): key: clusterID , value: samples index list
# - SOIs (np.ndarray[float]): normalised counts
# Return:
# - Returns an all zeroes float array, adapted for
# storing the logOdds for each type of copy number.
# dim= NbExons x [NbSOIs x [CN0, CN1, CN2,CN3+]]
def allocateLogOddsArray(SOIs, exons):
    # order=F should improve performance
    return (np.zeros((len(exons), (len(SOIs) * 4)), dtype=np.float, order='F'))


#####################################
# CNCalls:
#
# Args:
#
#
#
# Returns
#
def CNCalls(sex2Clust, exons, countsNorm, clusts2Samps, clusts2Ctrls, priors, SOIs, plotDir, logOddsArray):
    # Fixed Parameter
    bandwidth = 2

    #
    pdfFile = os.path.join(plotDir, "ResCallsByCluster_" + str(len(SOIs)) + "samps.pdf")
    # create a matplotlib object and open a pdf
    PDF = matplotlib.backends.backend_pdf.PdfPages(pdfFile)

    for clustID in clusts2Samps:
        ##################################
        ## Select cluster specific indexes in countsNorm
        ##### ROW indexes:
        # in case there are specific autosome and gonosome clusters.
        # identification of the indexes of the exons associated with the gonosomes or autosomes.
        if sex2Clust:
            (gonoIndex, genderInfo) = mageCNV.genderDiscrimination.getGenderInfos(exons)
            gonoIndexFlat = np.unique([item for sublist in list(gonoIndex.values()) for item in sublist])
            if clustID in sex2Clust["A"]:
                exonsIndex2Process = [i for i in range(countsNorm.shape[0]) if i not in gonoIndex]
            else:
                exonsIndex2Process = gonoIndex
        else:
            exonsIndex2Process = range(countsNorm.shape[0])

        ##### COLUMN indexes:
        # Get the indexes of the samples in the cluster and its controls
        sampleIndex2Process = clusts2Samps[clustID]
        if clustID in clusts2Ctrls:
            for controls in clusts2Ctrls[clustID]:
                sampleIndex2Process.extend(clusts2Samps[controls])
        sampleIndex2Process = list(set(sampleIndex2Process))

        # count data selection
        countsInCluster = np.take(countsNorm, exonsIndex2Process, axis=0)
        countsInCluster = np.take(countsInCluster, sampleIndex2Process, axis=1)

        ###########
        # Initialize InfoList with the exon index
        infoList = [[exon] for exon in exonsIndex2Process]

        ##################################
        # fit a gamma distribution to find the profile of exons with little or no coverage (CN0)
        # - gammaParameters
        # - gammaThreshold
        gammaParameters, gammaThreshold = fitGammaDistributionPrivate(countsInCluster, clustID, PDF)

        ###################################
        # Iterate over the exons
        for exon in range(len(exonsIndex2Process)):
            # Print progress every 10000 exons
            if exon % 10000 == 0:
                logger.info("%s: %s  %s ", clustID, exon, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

            # Get count data for the exon
            exonFPM = countsInCluster[exon]

            ####################
            # Filter n°1: the exon is not covered => mu == 0
            # do nothing leave logOdds at zero
            mean_fpkm = exonFPM.mean()
            if mean_fpkm == 0:
                infoList[exon] += [-1, 0, 0, 0]
                continue

            ###################
            # Fit a robust Gaussian to the count data
            # - mean:
            # - stdev:
            mean, stdev = fitRobustGaussianPrivate(exonFPM, bandwidth=bandwidth)

            ###################
            # Filter n°2: if standard deviation is zero
            # do nothing leave logOdds at zero
            if stdev == 0:
                # Define a new standard deviation to allow the calculation of the ratio
                if mean > gammaThreshold:
                    stdev = mean / 20
                # Exon nocall
                else:
                    infoList[exon] += [mean, -1, 0, 0]
                    continue

            z_score = (mean - gammaThreshold) / stdev
            weight = computeWeightPrivate(exonFPM, mean, stdev)

            ###################
            # Filter n°3:
            # Exon nocall
            if (weight < 0.5) or (z_score < 3):
                infoList[exon] += [mean, stdev, z_score, weight]
                continue

            # Append values to InfoList
            infoList[exon] += [mean, stdev, z_score, weight]

            # Retrieve results for each sample XXXXXX
            for i in clusts2Samps[clustID]:
                sample_data = exonFPM[sampleIndex2Process.index(i)]
                sampIndexInLogOddsArray = i * 4

                log_odds = mageCNV.copyNumbersCalls.computeLogOddsPrivate(sample_data, gammaParameters, gammaThreshold, priors, mean, stdev)

                for val in range(4):
                    if logOddsArray[exonsIndex2Process[exon], (sampIndexInLogOddsArray + val)] == 0:
                        logOddsArray[exonsIndex2Process[exon], (sampIndexInLogOddsArray + val)] = log_odds[val]
                    else:
                        logger.error('erase previous logOdds value')

        filtersPiePlotPrivate(clustID, infoList, PDF)

    # close the open pdf
    PDF.close()
    return(logOddsArray)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

############################
# fitGammaDistributionPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Estimate the parameters of the gamma distribution that best fits the data.
# The gamma distribution was chosen after testing 101 continuous distribution laws,
#  -it has few parameters (3 in total: shape, scale=1/beta, and loc),
#  -is well-known, and had the best goodness of fit on the empirical data.
# Arg:
# -countsInCluster (np.ndarray[floats]): cluster fragment counts (normalised)
# - clustID (str): cluster identifier
# - PDF (matplotlib object): store plots in a single pdf
# Returns a tupple (gamma_parameters, threshold_value), each variable is created here:
# -gammaParameters (tuple of floats): estimated parameters of the gamma distribution
# -thresholdValue (float): value corresponding to 95% of the cumulative distribution function
#
def fitGammaDistributionPrivate(countsInCluster, clustID, PDF):
    # compute meanFPM by exons
    # save computation time instead of taking the raw data (especially for clusters with many samples)
    meanCountByExons = np.mean(countsInCluster, axis=1)

    # smooth the coverage profile with kernel-density estimate using Gaussian kernels
    # - binEdges (np.ndarray[floats]): FPM range
    # - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
    #   dim= len(binEdges)
    binEdges, densityOnFPMRange = mageCNV.slidingWindow.smoothingCoverageProfile(meanCountByExons)

    # recover the threshold of the minimum density means before an increase
    # - minIndex (int): index from "densityMeans" associated with the first lowest
    # observed mean
    # - minMean (float): first lowest observed mean (not used for the calling step)
    (minIndex, minMean) = mageCNV.slidingWindow.findLocalMin(densityOnFPMRange)

    countsExonsNotCovered = meanCountByExons[meanCountByExons <= binEdges[minIndex]]

    countsExonsNotCovered.sort()  # sort data in-place

    # estimate the parameters of the gamma distribution that best fits the data
    gammaParameters = st.gamma.fit(countsExonsNotCovered)

    # compute the cumulative distribution function of the gamma distribution
    cdf = st.gamma.cdf(countsExonsNotCovered, a=gammaParameters[0], loc=gammaParameters[1], scale=gammaParameters[2])

    # find the index of the last element where cdf < 0.95
    thresholdIndex = np.where(cdf < 0.95)[0][-1]

    # compute the value corresponding to 95% of the cumulative distribution function
    # this value corresponds to the FPM value allowing to split covered exons from uncovered exons
    thresholdValue = countsExonsNotCovered[thresholdIndex]

    coverageProfilPlotPrivate(clustID, binEdges, densityOnFPMRange, minIndex, gammaParameters, PDF)

    return (gammaParameters, thresholdValue)


############################
# fitRobustGaussianPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Fits a single principal gaussian component around a starting guess point
# in a 1-dimensional gaussian mixture of unknown components with EM algorithm
# script found to :https://github.com/hmiemad/robust_Gaussian_fit (v07.2022)
# Args:
#  -X (np.ndarray[float]): A sample of 1-dimensional mixture of gaussian random variables
#  -mu (float, optional): Expectation. Defaults to None.
#  -sigma (float, optional): Standard deviation. Defaults to None.
#  -bandwidth (float, optional): Hyperparameter of truncation. Defaults to 1.
#  -eps (float, optional): Convergence tolerance. Defaults to 1.0e-5.
# Returns a tupple (mu,sigma), each variable is created here:
# mean [float] and stdev[float] of the gaussian component
def fitRobustGaussianPrivate(X, mu=None, sigma=None, bandwidth=1.0, eps=1.0e-5):
    # Integral of a normal distribution from -x to x
    weights = lambda x: erf(x / np.sqrt(2))
    # Standard deviation of a truncated normal distribution from -x to x
    sigmas = lambda x: np.sqrt(1 - 2 * x * st.norm.pdf(x) / weights(x))

    w, w0 = 0, 2

    if mu is None:
        # median is an approach as robust and naïve as possible to Expectation
        mu = np.median(X)

    if sigma is None:
        # rule of thumb
        sigma = np.std(X) / 3

    bandwidth_truncated_normal_weight = weights(bandwidth)
    bandwidth_truncated_normal_sigma = sigmas(bandwidth)

    while abs(w - w0) > eps:
        # loop until tolerence is reached
        try:
            """
            -create a window on X around mu of width 2*bandwidth*sigma
            -find the mean of that window to shift the window to most expected local value
            -measure the standard deviation of the window and divide by the standard
            deviation of a truncated gaussian distribution
            -measure the proportion of points inside the window, divide by the weight of
            a truncated gaussian distribution
            """
            W = np.where(np.logical_And(X - mu - bandwidth * sigma <= 0, X - mu + bandwidth * sigma >= 0), 1, 0)
            mu = np.mean(X[W == 1])
            sigma = np.std(X[W == 1]) / bandwidth_truncated_normal_sigma
            w0 = w
            w = np.mean(W) / bandwidth_truncated_normal_weight

        except:
            break

    return (mu, sigma)


############################
# computeWeightPrivate[PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# compute the sample contribution to the Gaussian obtained in a robust way.
#
# Args:
# - fpm_in_exon (np.ndarray[float]): FPM values for a particular exon for each sample
# - mean (float): mean FPM value for the exon
# - standard_deviation (float): std FPM value for the exon
# Returns weight of sample contribution to the gaussian for the exon [float]
def computeWeightPrivate(fpm_in_exon, mean, standard_deviation):
    targetData = fpm_in_exon[(fpm_in_exon > (mean - (2 * standard_deviation))) &
                             (fpm_in_exon < (mean + (2 * standard_deviation))), ]
    weight = len(targetData) / len(fpm_in_exon)

    return weight


############################
# computeLogOddsPrivate [PRIVATE FUNCTION, DO NOT CALL FROM OUTSIDE]
# Given four models, the log odds ratio (LOR) allows to choose the best-fitting model.
# Use of Bayes' theorem to deduce it.
#
# Args:
# - sample_data (float): a sample data point, FPM value
# - gamma_threshold (float):
# - prior_probabilities (list[float]): prior probabilities for different cases
# - mean (float): mean value for the normal distribution
# - standard_deviation (float): the standard deviation for the normal distribution
#
# Returns:
# - log_odds (list[float]): log-odds ratios for each copy number (CN0,CN1,CN2,CN3+)
def computeLogOddsPrivate(sample_data, params, gamma_threshold, prior_probabilities, mean, standard_deviation):
    # CN2 mean shift to get CN1 mean
    mean_cn1 = mean / 2

    # To Fill
    # Initialize an empty list to store the probability densities
    probability_densities = []

    ###############
    # Calculate the probability density for the gamma distribution (CN0 profil)
    # This is a special case because the gamma distribution has a heavy tail,
    # which means that the probability of density calculated from it can override
    # the other Gaussian distributions.
    # A condition is set up to directly associate a value of pdf to 0 if the sample FPM value
    # is higher than the mean of the Gaussian associated to CN1.
    # Reversely, the value of the pdf is truncated from the threshold value discriminating
    # covered from uncovered exons.
    gamma_pdf = 0
    cdf_cno_threshold = st.gamma.cdf(gamma_threshold, a=params[0], loc=params[1], scale=params[2])
    if sample_data <= mean_cn1:
        gamma_pdf = (1 / (1 - cdf_cno_threshold)) * st.gamma.pdf(sample_data, a=params[0], loc=params[1], scale=params[2])
    probability_densities.append(gamma_pdf)

    ################
    # Calculate the probability densities for the remaining cases (CN1,CN2,CN3+) using the normal distribution
    probability_densities.append(st.norm.pdf(sample_data, mean / 2, standard_deviation))
    probability_densities.append(st.norm.pdf(sample_data, mean, standard_deviation))
    probability_densities.append(st.norm.pdf(sample_data, 3 * mean / 2, standard_deviation))

    #################
    # Calculate the prior probabilities
    probability_densities_priors = []
    for i in range(len(probability_densities)):
        probability_densities_priors.append(probability_densities[i] * prior_probabilities[i])

    ##################
    # Calculate the log-odds ratios
    log_odds = []
    for i in range(len(probability_densities_priors)):
        # Calculate the denominator for the log-odds ratio
        to_subtract = probability_densities_priors[:i] + probability_densities_priors[i + 1:]
        to_subtract = np.sum(to_subtract)

        # Calculate the log-odds ratio for the current probability density
        if np.isclose(np.log10(to_subtract), 0):
            log_odd = 0
        else:
            log_odd = np.log10(probability_densities_priors[i]) - np.log10(to_subtract)

        log_odds.append(log_odd)

    return log_odds


###################################
# coverageProfilPlotPrivate:
# generates a plot per cluster
# x-axis: the range of FPM bins (every 0.1 between 0 and 10)
# y-axis: exons densities
# black curve: density data smoothed with kernel-density estimate using Gaussian kernels
# red vertical line: minimum FPM threshold, all uncovered exons are below this threshold
# green curve: gamma fit
#
# Args:
# - sampleName (str): sample exact name
# - binEdges (np.ndarray[floats]): FPM range
# - densityOnFPMRange (np.ndarray[float]): probability density for all bins in the FPM range
#   dim= len(binEdges)
# - minIndex (int): index associated with the first lowest density observed
# - gammaParameters (list[float]):
# - pdf (matplotlib object): store plots in a single pdf
#
# save a plot in the output pdf
def coverageProfilPlotPrivate(clustID, binEdges, densityOnFPMRange, minIndex, gammaParameters, PDF):

    fig = plt.figure(figsize=(6, 6))
    plt.plot(binEdges, densityOnFPMRange, color='black', label='smoothed densities')

    pdfCN0 = st.gamma.pdf(binEdges, a=gammaParameters[0], loc=gammaParameters[1], scale=gammaParameters[2])
    plt.plot(binEdges, pdfCN0, 'c', label=("CN0 α=" + str(round(gammaParameters[0], 2)) +
                                           " loc=" + str(round(gammaParameters[1], 2)) +
                                           " β=" + str(round(gammaParameters[2], 2))))
    plt.axvline(binEdges[minIndex], color='crimson', linestyle='dashdot', linewidth=2,
                label="minFPM=" + '{:0.1f}'.format(binEdges[minIndex]))

    plt.ylim(0, 0.5)
    plt.ylabel("Exon densities")
    plt.xlabel("Fragments Per Million")
    plt.title(clustID + " coverage profile")
    plt.legend()

    PDF.savefig(fig)
    plt.close()


###################################
# filtersPiePlotPrivate:
# generates a plot per cluster
#
# Args:
# - clustID [str]:
# - infoList (list of list[float]):
# - pdf (matplotlib object): store plots in a single pdf
#
# save a plot in the output pdf
def filtersPiePlotPrivate(clustID, infoList, pdf):

    fig = plt.figure(figsize=(10, 10))

    exonsMuZero = len(infoList[infoList[0] == -1])
    exonsSigRGZero = len(infoList[infoList[1] == -1])
    exonsZscore_inf3_only = len(infoList[(infoList[2] < 3) & infoList[3] >= 0.50])
    exonsZscore_Weigth = len(infoList[(infoList[2] < 3) & infoList[3] < 0.50])
    exonsWeight_inf_50p = len(infoList[(infoList[2] >= 3) & infoList[3] < 0.50])
    exonsToKeep = len(infoList[(infoList[0] > 0) & (infoList[1] > 0) & (infoList[2] >= 3) & infoList[3] > 0.50])

    x = [exonsMuZero, exonsSigRGZero, exonsZscore_inf3_only, exonsZscore_Weigth,
         exonsWeight_inf_50p, exonsToKeep]

    plt.pie(x, labels=['exons filtered mu=0', 'exons filtered sigRG=0', 'exons filtered only Zscore <3',
                       'exons filtered Zscore+Weight', 'exons filtered Weight <50%', 'exons Keep'],
            colors=["grey", "yellow", "indianred", "mediumpurple", "royalblue", "mediumaquamarine"],
            autopct=lambda x: str(round(x, 2)) + '%',
            startangle=-270,
            pctdistance=0.7, labeldistance=1.1)
    plt.legend()
    plt.title(clustID)

    pdf.savefig(fig)
    plt.close()