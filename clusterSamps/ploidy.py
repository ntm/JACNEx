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
import callCNVs.likelihoods
import countFrags.bed

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################################################
# Estimate the ploidy of each chromosome for each sample in a VALID cluster.
# Print results in TSV format to ploidyFile in TSV format, columns:
# SAMPLE CLUSTER_A CLUSTER_G ANEUPLOIDIES [one column per chromosome]
# For each sample, we store the fraction of reads that map to each chromosome;
# for each cluster and each chrom, we calculate the mean and stddev of these fractions;
# finally for each sample, any chrom where it's fraction is beyond mu +- 3*sigma
# for its cluster (autosome or gonosome, depending on the chrom) is shown in the
# ANEUPLOIDIES column with the ratio: fraction/mu .
#
# Args:
# - *FPMs, *exons, samples: as returned by countFrags.countsFile.parseAndNormalizeCounts()
# - clust2samps, fitWith, clustIsValid: as returned by clusterSamps.clustering.buildClusters()
# - wgsCN0sigma: for WGS data, passed by s2_clusterSamps.py
# - ploidyFile: filename (with path) where results will be saved, must not exist.
#
# Return nothing.
def estimatePloidy(autosomeFPMs, gonosomeFPMs, intergenicFPMs, autosomeExons, gonosomeExons,
                   samples, clust2samps, fitWith, clustIsValid, wgsCN0sigma, ploidyFile):
    ########################################
    # sanity checks
    nbExonsA = autosomeFPMs.shape[0]
    nbExonsG = gonosomeFPMs.shape[0]
    nbSamples = len(samples)
    if nbExonsA != len(autosomeExons):
        logger.error("sanity check nbExonsAutosomes, impossible!")
        raise Exception("estimatePloidy() sanity check failed")
    if nbExonsG != len(gonosomeExons):
        logger.error("sanity check nbExonsGonosomes, impossible!")
        raise Exception("estimatePloidy() sanity check failed")
    if (nbSamples != autosomeFPMs.shape[1]) or (nbSamples != gonosomeFPMs.shape[1]):
        logger.error("sanity check nbSamples, impossible!")
        raise Exception("estimatePloidy() sanity check failed")

    ########################################
    # Prep useful data structures for later
    # samp2index: dict, key==sampleID, value==index in samples
    samp2index = {}
    for si in range(nbSamples):
        samp2index[samples[si]] = si

    # samp2clustA, samp2clustG: dicts, key==sampleID, value==autosome/gonosome clusterID
    samp2clustA = {}
    samp2clustG = {}
    for clust in clust2samps.keys():
        if clust.startswith('A_'):
            samp2clust = samp2clustA
        elif clust.startswith('G_'):
            samp2clust = samp2clustG
        else:
            logger.error("clusterID %s doesn't start with A_ or G_, fix the code!")
            raise('bad clusterID')
        for samp in clust2samps[clust]:
            if samp in samp2clust:
                logger.error("sample %s is in more than one gonosome cluster, impossible!")
                raise('sanity-check failed')
            samp2clust[samp] = clust

    # FPM cut-off to characterize exons that aren't captured - use cutoff provided by fitCN0(),
    # ignoring first returned value (CN0sigma)
    maxFPMuncaptured = callCNVs.likelihoods.fitCN0(intergenicFPMs, wgsCN0sigma)[1]
    # sex chromosomes and their type
    sexChroms = countFrags.bed.sexChromosomes()

    ########################################
    # fill sumOfFPMs for each chrom
    # key == chrom, value == numpy 1D-array of size nbSamples (in the same order as samples)
    sumOfFPMs = {}
    # along the way, collect the autosome/gonosome chrom names, same order as in *exons
    chromsA = []
    chromsG = []
    # code is the same for autosomes and gonosomes => we use chromType, autosome then gonosome
    for chromType in range(2):
        if chromType == 0:
            FPMs = autosomeFPMs
            exons = autosomeExons
            chroms = chromsA
        else:
            FPMs = gonosomeFPMs
            exons = gonosomeExons
            chroms = chromsG

        thisChrom = exons[0][0]
        chroms.append(thisChrom)
        firstExonIndex = 0
        for ei in range(len(exons) + 1):
            if (ei == len(exons)) or (exons[ei][0] != thisChrom):
                # we are finished or changing chrom -> process thisChrom
                # FPMs for thisChrom, using a slice so we get a view
                theseFPMs = FPMs[firstExonIndex:ei, :]
                # "accepted" exons on thisChrom: exons that are "captured" (FPM > maxFPMuncaptured) in
                # "most" samples (at least 80%, hard-coded as 0.2 below).
                # This provides a cleaner signal when samples use different capture kits, but don't
                # apply this strat to chrY...
                if (chromType == 0) or (sexChroms[thisChrom] == 1):
                    twentyPercentQuantilePerExon = numpy.quantile(theseFPMs, 0.2, axis=1)
                    sumOfFPMs[thisChrom] = numpy.sum(theseFPMs[twentyPercentQuantilePerExon > maxFPMuncaptured, :],
                                                     axis=0)
                else:
                    # chrY|W: no 20%-quantile filter
                    sumOfFPMs[thisChrom] = numpy.sum(theseFPMs, axis=0)

                # ok move on to next chrom (except if we are finished)
                if (ei != len(exons)):
                    thisChrom = exons[ei][0]
                    chroms.append(thisChrom)
                    firstExonIndex = ei

    ########################################
    # process sumOfFPMs:
    # for each VALID cluster, for each chrom, calculate mean and stddev of its samples' FPMs
    # clust2chrom2stats: dict, key==clustID, value==dict with key==chrom, value==tuple (mean, stddev)
    clust2chrom2stats = {}
    for clust in clust2samps.keys():
        # skip INVALID clusters
        if not clustIsValid[clust]:
            continue
        if clust.startswith('A_'):
            chroms = chromsA
        else:
            chroms = chromsG
        clust2chrom2stats[clust] = {}
        # sampInClust: numpy array of nbSamples bools, True iff sample is in clust
        sampInClust = numpy.zeros(nbSamples, dtype=bool)
        for samp in clust2samps[clust]:
            sampInClust[samp2index[samp]] = True
        for chrom in chroms:
            sumsThisChrom = sumOfFPMs[chrom][sampInClust]
            mu = numpy.mean(sumsThisChrom)
            sigma = numpy.std(sumsThisChrom, mean=mu)
            clust2chrom2stats[clust][chrom] = (mu, sigma)

    ########################################
    # print results to ploidyFile, calling aneuploidies as we go
    try:
        outFH = open(ploidyFile, "x")
    except Exception as e:
        logger.error("Cannot open ploidyFile %s: %s", ploidyFile, e)
        raise Exception('cannot open ploidyFile')

    toPrint = "SAMPLE\tCLUSTER_A\tCLUSTER_G\tANEUPLOIDIES\t" + "\t".join(chromsA + chromsG) + "\n"
    outFH.write(toPrint)

    for samp in samples:
        toPrint = samp
        aneupl = []
        sumsThisSamp = ""
        for chromType in range(2):
            if chromType == 0:
                clust = samp2clustA[samp]
                chroms = chromsA
            else:
                clust = samp2clustG[samp]
                FPMs = gonosomeFPMs
                chroms = chromsG

            if clustIsValid[clust]:
                toPrint += "\t" + clust
                for chrom in chroms:
                    (mu, sigma) = clust2chrom2stats[clust][chrom]
                    thisSumOfFPMs = sumOfFPMs[chrom][samp2index[samp]]
                    if (thisSumOfFPMs < mu - 3 * sigma) or (thisSumOfFPMs > mu + 3 * sigma):
                        FPMratio = thisSumOfFPMs / mu
                        aneupl.append(chrom + ":%.2f" % FPMratio)
            else:
                toPrint += "\t" + clust + " (INVALID)"

            # we print the sums of FPMs per chrom, whether cluster is valid or not
            for chrom in chroms:
                sumsThisSamp += "\t%.0f" % sumOfFPMs[chrom][samp2index[samp]]
        toPrint += "\t" + ','.join(aneupl)
        toPrint += sumsThisSamp + "\n"
        outFH.write(toPrint)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
