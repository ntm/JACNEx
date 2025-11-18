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


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################


#############################################################
# Estimate the ploidy of each chromosome in each sample:
# for each sample, count the sum of FPMs that map to exons on each chrom;
# for each cluster and each chrom, calculate the mean and stddev of these sums,
#   including samples in fitWith clusters;
# finally for each sample, we call an aneuploidy for any chrom where
#   |sumOfFPMs - mu| / mu > aneuplMinShift AND
#   |sumOfFPMs - mu| / sigma > aneuplMinZscore
#
# Print results to ploidyFile in TSV format, columns:
# SAMPLE GENDER CLUSTER_A CLUSTER_G ANEUPLOIDIES [one column per chromosome]
# Per-sample gender predictions in the GENDER column come straight from samp2gender,
# ie they are not called here.
# Called aneuploidies are shown in the ANEUPLOIDIES column, with the relative shift
# of sumOfFPMs and the Z-score.
# If a cluster + its fitWiths contain in total less than minPloidySamps, stats are
# meaningless -> its samples get "ANEUPL NOT CALLED" (and aneuploidies don't get called).
# However a cluster can be INVALID because too small to call CNVs, but still large
# enough (>= minPloidySamps) to get reasonable aneuploidy calls.
# Args:
# - *FPMs, *exons, samples: as returned by countFrags.countsFile.parseAndNormalizeCounts()
# - clust2samps, clustIsValid: as returned by clusterSamps.clustering.buildClusters()
# - ploidyFile: filename (with path) where results will be saved, must not exist.
#
# Return nothing.
def estimatePloidy(autosomeFPMs, gonosomeFPMs, autosomeExons, gonosomeExons, samples,
                   clust2samps, fitWith, clustIsValid, samp2gender, clust2gender, ploidyFile):
    # hard-coded cutoffs for calling aneuploidies, see head-of-function comments
    aneuplMinShift = 0.25
    aneuplMinZscore = 3
    minPloidySamps = 5
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

    ########################################
    # fill sumOfFPMs for each chrom
    # key == chrom, value == numpy 1D-array of size nbSamples (in the same order as samples)
    sumOfFPMs = {}
    # along the way, collect the autosome/gonosome chrom names (excluding Mito), same order as in *exons
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
                sumOfFPMs[thisChrom] = numpy.sum(theseFPMs, axis=0)

                # ok move on to next chrom (except if we are finished)
                if (ei != len(exons)):
                    thisChrom = exons[ei][0]
                    chroms.append(thisChrom)
                    firstExonIndex = ei

    ########################################
    # process sumOfFPMs:
    # for each cluster (>= minPloidySamps), for each chrom, analyze its
    # samples' FPMs (including fitWith samples if any) in two passes:
    # pass 1: calculate stddev, make sure at least 75% of samples are
    #     within 3 stddevs of the median (else give up on this chrom),
    #     and calculate the new stddev restricted to these samples;
    # pass 2: restrict samples to those within 3 stddevs of the median
    #     and calculate the new (mu,stddev) restricted to these samples,
    #     saving this in clust2chrom2stats.
    # clust2chrom2stats: dict, key==clustID, value==dict with key==chrom, value==tuple (mean, stddev)
    clust2chrom2stats = {}
    for clust in clust2samps.keys():
        if clust.startswith('A_'):
            chroms = chromsA
        else:
            chroms = chromsG
        # sampInClust: numpy array of nbSamples bools, True iff sample is in clust or in a fitWith
        sampInClust = numpy.zeros(nbSamples, dtype=bool)
        sampsInClust = 0
        for cfw in [clust] + fitWith[clust]:
            for samp in clust2samps[cfw]:
                sampInClust[samp2index[samp]] = True
                sampsInClust += 1
        # cannot call anything on tiny clusters
        if (sampsInClust < minPloidySamps):
            continue
        clust2chrom2stats[clust] = {}
        for chrom in chroms:
            sumsThisChrom = sumOfFPMs[chrom][sampInClust]
            median = numpy.median(sumsThisChrom)
            sigma = numpy.std(sumsThisChrom)
            sumsRestricted = sumsThisChrom[numpy.abs(sumsThisChrom - median) < 3 * sigma]
            if sumsRestricted.size < sampsInClust * 0.75:
                logger.warning("cluster %s chromosome %s: cannot call ploidy, FPMs too far from monomodal",
                               clust, chrom)
                continue
            mu = numpy.mean(sumsRestricted)
            # sigma = numpy.std(sumsRestricted, mean=mu)
            # mean= only available starting at numpy 2.0.0
            sigmaRestricted = numpy.std(sumsRestricted)
            clust2chrom2stats[clust][chrom] = (mu, sigmaRestricted)
            logger.debug("cluster %s chrom %s: mu=%.0f sigma=%.0f", clust, chrom, mu, sigma)

    ########################################
    # print results to ploidyFile, calling aneuploidies as we go
    try:
        outFH = open(ploidyFile, "x")
    except Exception as e:
        logger.error("Cannot open ploidyFile %s: %s", ploidyFile, e)
        raise Exception('cannot open ploidyFile')

    toPrint = "SAMPLE\tGENDER\tCLUSTER_A\tCLUSTER_G\tANEUPLOIDIES (chrom:FPM-relative-shift:Z-score)\t" + "\t".join(chromsA + chromsG) + "\n"
    outFH.write(toPrint)

    for samp in samples:
        toPrint = samp + "\t" + samp2gender[samp]
        aneupl = []
        sumsThisSamp = ""
        for chromType in range(2):
            if chromType == 0:
                clust = samp2clustA[samp]
                chroms = chromsA
            else:
                clust = samp2clustG[samp]
                chroms = chromsG

            toPrint += "\t" + clust
            if chromType == 1:
                toPrint += " (" + clust2gender[clust] + ")"
            if fitWith[clust]:
                toPrint += " FITWITH"
                for fw in fitWith[clust]:
                    toPrint += " " + fitWith[clust]
                    if chromType == 1:
                        toPrint += " (" + clust2gender[fw] + ")"
            if not clustIsValid[clust]:
                toPrint += " (INVALID)"
            if clust not in clust2chrom2stats:
                toPrint += " (ANEUPL NOT CALLED)"
            else:
                for chrom in chroms:
                    # don't call aneuploidies on Mito
                    if (chrom == 'M') or (chrom == 'MT') or (chrom == 'chrM') or (chrom == 'chrMT'):
                        continue
                    # also can't call if FPMs were not monomodal at all for this chrom
                    if (chrom not in clust2chrom2stats[clust]):
                        continue
                    (mu, sigma) = clust2chrom2stats[clust][chrom]
                    thisSumOfFPMs = sumOfFPMs[chrom][samp2index[samp]]
                    absDiff = abs(thisSumOfFPMs - mu)
                    if (absDiff / mu > aneuplMinShift) and (absDiff / sigma > aneuplMinZscore):
                        FPMratio = thisSumOfFPMs / mu
                        aneupl.append(chrom + ":%.2f:%.1f" % (FPMratio, absDiff / sigma))

            # we print the sums of FPMs per chrom, even if we couldn't call aneuploidies
            for chrom in chroms:
                sumsThisSamp += "\t%.0f" % sumOfFPMs[chrom][samp2index[samp]]
        toPrint += "\t" + ','.join(aneupl)
        toPrint += sumsThisSamp + "\n"
        outFH.write(toPrint)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################
