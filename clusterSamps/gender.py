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

####### JACNEx modules
import countFrags.bed

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###############################################################################
# Predict the gender of each sample (ie column of exonsFPM) based on chrX and chrY FPMs.
#
# Args:
# - FPMs: numpy.ndarray of FPMs for exons on gonosomes, size = nbExons x nbSamples
#
# Return samp2gender: key == sampleID, value=='M' or 'F' or 'X0' or 'XXY'
def predictGenders(FPMs, exons, samples):
    # sanity
    nbExons = FPMs.shape[0]
    nbSamples = FPMs.shape[1]
    if nbExons != len(exons):
        logger.error("sanity check nbExons, impossible!")
        raise Exception("predictGenders() sanity check failed")
    if nbSamples != len(samples):
        logger.error("sanity check nbSamples, impossible!")
        raise Exception("predictGenders() sanity check failed")

    # exonOnY: numpy array of nbExons bools, exonOnY[ei]==False if exons[ei] is
    # on chrX/Z, True if on chrY/W
    exonOnY = numpy.zeros(nbExons, dtype=bool)
    sexChroms = countFrags.bed.sexChromosomes()
    for ei in range(nbExons):
        if sexChroms[exons[ei][0]] == 2:
            exonOnY[ei] = True
    YWexonsFPMs = FPMs[exonOnY, :]
    sumOfFPMsYW = numpy.sum(YWexonsFPMs, axis=0)
    XZexonsFPMs = FPMs[numpy.logical_not(exonOnY), :]
    sumOfFPMsXZ = numpy.sum(XZexonsFPMs, axis=0)

    ########################################
    # samp2gender: key == sampleID, value=='M' or 'F' or 'X0' or 'XXY'
    samp2gender = {}
    # Unsupervised clustering (e.g. K-means) feels good but it won't work when
    # genders are strongly imbalanced...
    # Instead we start with an empirical method on chrY, and then make sure
    # chrX agrees

    # find the first "large" (> largeGapSize) gap in FPMs on chrY.
    # largeGapSize is set empirically, it works for all the data we tested but
    # if predictGenders() fails we may need a different value (or method!)
    largeGapSize = 150
    fpmThreshold = 0
    prevFPM = 0
    sumOfFPMsYWsorted = numpy.sort(sumOfFPMsYW)
    for fpm in sumOfFPMsYWsorted:
        if fpm - prevFPM > largeGapSize:
            fpmThreshold = (fpm + prevFPM) / 2
            break
        else:
            prevFPM = fpm
    # if all clusters have "high" FPMs on chrY: all-male? or largeGapSize is too small?
    if largeGapSize < sumOfFPMsYWsorted[0]:
        logger.warning("all samples are predicted to be Male based on chrY, is this plausible?")
        logger.warning("If not, please let us know so we can fix it.")
    elif fpmThreshold == 0:
        # didn't find a large gap: all-female (or very low coverage of chrY, or...)
        logger.warning("all samples are predicted to be Female based on chrY, is this plausible?")
        logger.warning("If not, please let us know so we can fix it.")
        fpmThreshold = sumOfFPMsYWsorted[-1] + 1

    # assign genders based on chrY
    for si in range(nbSamples):
        if sumOfFPMsYW[si] <= fpmThreshold:
            samp2gender[samples[si]] = 'F'
        else:
            samp2gender[samples[si]] = 'M'

    if (largeGapSize < sumOfFPMsYWsorted[0]) or (fpmThreshold > sumOfFPMsYWsorted[-1]):
        # all samples are same gender, cannot do anything more
        return(samp2gender)

    # we have both M and F samples according to chrY, make sure chrX agrees:
    # indexes of 'F' samples, sorted by increasing FPMs on chrX
    femalesByFPMonX = []
    # and indexes of 'M' samples sorted by decreasing FPMs on chrX
    malesByFPMonX = []
    for si in range(nbSamples):
        if samp2gender[samples[si]] == 'F':
            femalesByFPMonX.append(si)
        else:
            malesByFPMonX.append(si)
    femalesByFPMonX.sort(key=lambda si: sumOfFPMsXZ[si])
    malesByFPMonX.sort(key=lambda si: sumOfFPMsXZ[si], reverse=True)

    smallestFi = 0
    largestMi = 0
    # numbers of called X0 and XXY
    countX0 = 0
    countXXY = 0
    while(sumOfFPMsXZ[femalesByFPMonX[smallestFi]] < sumOfFPMsXZ[malesByFPMonX[largestMi]]):
        # count number of M whose FPMonX is greater than smallestFi
        nbMabove = 1
        for mi in range(largestMi + 1, len(malesByFPMonX)):
            if femalesByFPMonX[smallestFi] < malesByFPMonX[mi]:
                nbMabove += 1
            else:
                break
        # and number of F whose FPMonX is smaller than largestMi
        nbFbelow = 1
        for fi in range(smallestFi + 1, len(femalesByFPMonX)):
            if femalesByFPMonX[fi] < malesByFPMonX[largestMi]:
                nbFbelow += 1
            else:
                break
        # if counts are equal, the bad sample is the one from the most frequent gender
        if nbMabove == nbFbelow:
            if len(femalesByFPMonX) < len(malesByFPMonX):
                nbFbelow += 1
            else:
                nbMabove += 1

        # the sample that disagrees with the most other-gender samples is bad
        if (nbMabove > nbFbelow):
            # 'F' sample seems to be X0 (Turner)
            samp2gender[samples[femalesByFPMonX[smallestFi]]] = 'X0'
            countX0 += 1
            smallestFi += 1
            if smallestFi >= len(femalesByFPMonX):
                # no more 'F' samples, we are done
                break
        else:
            # 'M' sample seems to be XXY (Klinefelter)
            samp2gender[samples[malesByFPMonX[largestMi]]] = 'XXY'
            countXXY += 1
            largestMi += 1
            if largestMi >= len(malesByFPMonX):
                # no more 'M' samples, we are done
                break

    if (countX0 + countXXY > 0):
        logMess = "sex chromosomal anomalies detected: "
        if (countX0 > 0):
            logMess += str(countX0) + " X0 samples "
        if (countX0 * countXXY > 0):
            logMess += "and "
        if (countXXY > 0):
            logMess += str(countXXY) + " XXY samples "
        logMess += "were called, examine the ploidy file"
        logger.warning(logMess)

    return(samp2gender)


###############################################################################
# Assign a gender for each cluster, using a majority vote.
# Log any disagreements between samples and their cluster.
# X0 and XXY samples are ignored for the vote, but if all samples in a cluster
# are X0 or XXY the cluster is assigned 'F'.
#
# Args:
# - samp2gender as returned by assignGender
#
# Return clust2gender: key == clusterID (gonosomes only), value=='M' or 'F'
def clusterGender(samp2gender, clust2samps, fitWith):
    # clust2gender: key == clusterID, value=='M' or 'F'
    clust2gender = {}

    for clust in clust2samps.keys():
        countF = 0
        countM = 0
        for samp in clust2samps[clust]:
            if samp2gender[samp] == 'F':
                countF += 1
            elif samp2gender[samp] == 'M':
                countM += 1
            # else X0 or XXY, NOOP
        if (countF >= countM):
            # this includes the case where all samples are X0 or XXY
            clust2gender[clust] = 'F'
        else:
            clust2gender[clust] = 'M'
        if (countF * countM > 0):
            logger.warning("cluster %s contains both M and F samples, this shouldn't happen! Please let us know",
                           clust)

    # make sure clusters got the same genders as their FITWITHs
    for clust in clust2gender.keys():
        gender = clust2gender[clust]
        for fw in fitWith[clust]:
            if (clust2gender[fw] != gender):
                logger.warning("cluster %s is FITWITH %s, but their genders are predicted differently: %s and %s",
                               clust, fw, gender, clust2gender[fw])
                logger.warning("CNV calls on the sex chromosomes will be lower quality. Please let us know so we can fix it.")

    return(clust2gender)
