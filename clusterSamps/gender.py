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
import numpy
import statistics

####### JACNEx modules
import countFrags.bed

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###############################################################################
# Assign a gender for each sample (ie column of exonsFPM)
#
# Args:
# - FPMs: numpy.ndarray of FPMs for exons on gonosomes, size = nbExons x nbSamples
#
# Return clust2gender: key == clusterID (gonosomes only), value=='M' or 'F'
#
# If we're unable to assign a gender to each sample, log a warning and return an
# all-female assignment vector. This can happen legitimately if the cohort is single-gender,
# but it could also result from noisy / heterogeneous data, or flaws in our methodology.
def assignGender(FPMs, exons, samples, clust2samps, fitWith):
    # sanity
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    if nbExons != len(exons):
        logger.error("sanity check nbExons, impossible!")
        raise Exception("assignGender() sanity check failed")
    if nbSamples != len(samples):
        logger.error("sanity check nbSamples, impossible!")
        raise Exception("assignGender() sanity check failed")

    # exonOnY: numpy array of nbExons bools, exonOnY[ei]==False if exons[ei] is
    # on chrX/Z, True if on chrY/W
    exonOnY = numpy.zeros(nbExons, dtype=bool)
    sexChroms = countFrags.bed.sexChromosomes()
    for ei in range(nbExons):
        if sexChroms[exons[ei][0]] == 2:
            exonOnY[ei] = True
    XZexonsFPMs = FPMs[numpy.logical_not(exonOnY), :]
    sumOfFPMsXZ = numpy.sum(XZexonsFPMs, axis=0)
    YWexonsFPMs = FPMs[exonOnY, :]
    sumOfFPMsYW = numpy.sum(YWexonsFPMs, axis=0)

    # samp2clust: dict, key==sampleID, value==clusterID
    samp2clust = {}
    for clust in clust2samps.keys():
        for samp in clust2samps[clust]:
            if samp in samp2clust:
                logger.error("sample %s is in more than one gonosome cluster, impossible!")
                raise('sanity-check failed')
            samp2clust[samp] = clust

    ########################################
    # clust2FPM_*: dict, key==clusterID, value == median (over all samples in the cluster) of
    # the sum of FPMs on this chrom
    clust2FPM_X = {}
    clust2FPM_Y = {}

    # start by building lists of sumOfFPMs, then calculate the median
    for clust in clust2samps.keys():
        clust2FPM_X[clust] = []
        clust2FPM_Y[clust] = []
    for si in range(nbSamples):
        clust = samp2clust[samples[si]]
        clust2FPM_X[clust].append(sumOfFPMsXZ[si])
        clust2FPM_Y[clust].append(sumOfFPMsYW[si])
    for clust in clust2samps.keys():
        clust2FPM_X[clust] = statistics.median(clust2FPM_X[clust])
        clust2FPM_Y[clust] = statistics.median(clust2FPM_Y[clust])

    ########################################
    # Predict the gender of each cluster (we assume that clusters are single-gender,
    # i.e. clustering didn't totally fail).
    # clust2gender: key == clusterID, value=='M' or 'F'
    clust2gender = {}
    # Unsupervised clustering (e.g. K-means) feels good but it won't work when
    # genders are strongly imbalanced...
    # Instead we start with an empirical method on chrY, and then make sure
    # chrX agrees

    # find the first "large" (> largeGapSize) gap in FPMs on chrY.
    # largeGapSize is set empirically, it works for all the data we tested but
    # if assignGender() fails we may need a different value (or method!)
    largeGapSize = 200
    fpmThreshold = 0
    medianFPMsY = sorted(clust2FPM_Y.values())
    prevFPM = 0
    for fpm in medianFPMsY:
        if fpm - prevFPM > largeGapSize:
            fpmThreshold = (fpm + prevFPM) / 2
            break
        else:
            prevFPM = fpm
    # if all clusters have "high" FPMs on chrY: all-male? or largeGapSize is too small?
    if largeGapSize < medianFPMsY[0]:
        logger.warning("all samples are predicted to be Male, is this correct? If not, CNV calls on the")
        logger.warning("sex chromosomes will be lower quality. Please let us know so we can fix it.")
    elif fpmThreshold == 0:
        # didn't find a large gap: all-female (or algo didn't work properly)
        logger.warning("all samples are predicted to be Female, is this correct? If not, CNV calls on the")
        logger.warning("sex chromosomes will be lower quality. Please let us know so we can fix it.")
        fpmThreshold = medianFPMsY[-1] + 1

    for clust in clust2FPM_Y.keys():
        if clust2FPM_Y[clust] <= fpmThreshold:
            clust2gender[clust] = 'F'
        else:
            clust2gender[clust] = 'M'

    # now make sure chrX agrees
    for clustM in clust2gender.keys():
        if clust2gender[clustM] != 'M':
            continue
        for clustF in clust2gender.keys():
            if clust2gender[clustM] != 'F':
                continue
            if clust2FPM_X[clustM] >= clust2FPM_X[clustF]:
                logger.warning("clusters %s and %s are predicted as Male and Female based on chrY, but chrX doesn't agree!",
                               clustM, clustF)
                logger.warning("CNV calls on the sex chromosomes will be lower quality. Please let us know so we can fix it.")

    # also make sure clusters got the same genders as their FITWITHs
    for clust in clust2gender.keys():
        gender = clust2gender[clust]
        for fw in fitWith[clust]:
            if (clust2gender[fw] != gender):
                logger.warning("cluster %s is FITWITH %s, but their genders are predicted differently: %s and %s",
                               clust, fw, gender, clust2gender[fw])
                logger.warning("CNV calls on the sex chromosomes will be lower quality. Please let us know so we can fix it.")

    return(clust2gender)
