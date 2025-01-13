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


import gzip
import logging
import re

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#####################################
# parseClustsFile
# Args:
# - clustsFile (str): a clustersFile produced by printClustsFile(), possibly gzipped
#
# A clustersFile is a TSV with columns CLUSTER_ID FIT_WITH GENDER VALID SAMPLES.
# The CLUSTER_IDs must be strings formatted as TYPE_NUMBER, where TYPE is 'A' or 'G'
#    depending on whether the clustering was performed using counts from exons on
#    autosomes or gonosomes (==sex chromosomes)
# FIT_WITH is a comma-separated list of clusterIDs;
# GENDER is 'M' or 'F' for Gonosome clusters (empty for Autosome clusters)
# VALID is 0 or 1, 0==invalid means the cluster fails some QC and should be ignored;
# SAMPLES is a comma-separated list of sampleIDs.
#
# The idea behind FIT_WITH: if cluster C1 is too small for a robust fit of the copy-number
# (CN2) distribution, rather than ignore all samples in C1, we fit CN2 using additional
# samples from other clusters that are hopefully similar enough to C1.
#
# Returns a tuple (clust2samps, samp2clusts, fitWith, clust2gender, clustIsValid)
# - clust2samps: dict, key==clusterID, value == list of sampleIDs
# - samp2clusts: dict, key==sampleID, value == list of clusterIDs (typically one cluster
#   of each TYPE), this is redundant with clust2samps and is provided for convenience
# - fitWith: dict, key==clusterID, value == list of clusterIDs (must be same TYPE)
# - clust2gender: dict, key==clusterID (Gonosome clusters only), value == string 'M' or 'F'
# - clustIsValid: dict, key==clusterID, value == Boolean
def parseClustsFile(clustsFile):
    try:
        if clustsFile.endswith(".gz"):
            clustsFH = gzip.open(clustsFile, "rt")
        else:
            clustsFH = open(clustsFile, "r")
    except Exception as e:
        logger.error("Opening provided clustsFile %s: %s", clustsFile, e)
        raise Exception('cannot open clustsFile')

    # To return
    clust2samps = {}
    samp2clusts = {}
    fitWith = {}
    clust2gender = {}
    clustIsValid = {}

    # regular expression for sanity-checking clusterIDs
    clustPattern = re.compile(r'^(A|G)_\d+$')

    # skip header
    clustsFH.readline()

    for line in clustsFH:
        try:
            (clusterID, fitWithThisClust, gender, valid, samples) = line.rstrip().split("\t")
        except Exception:
            logger.error("Provided clustsFile %s MUST have 5 tab-separated fields, it doesn't in line: %s", clustsFile, line)
            raise Exception('clustsFile bad line')

        if not clustPattern.match(clusterID):
            logger.error("In clustsFile %s, badly formatted clusterID %s", clustsFile, clusterID)
            raise Exception('clustsFile bad line')
        if clusterID in clust2samps:
            logger.error("In clustsFile %s, clusterID %s appears twice", clustsFile, clusterID)
            raise Exception('clustsFile bad line')

        samplesInClust = samples.split(',')
        clust2samps[clusterID] = samplesInClust
        for s in samplesInClust:
            if s not in samp2clusts:
                samp2clusts[s] = []
            samp2clusts[s].append(clusterID)

        if fitWithThisClust != '':
            fitWithList = fitWithThisClust.split(',')
            for f in fitWithList:
                if not clustPattern.match(f):
                    logger.error("In clustsFile %s for CLUSTER_ID=%s, badly formatted clusterID %s in FIT_WITH",
                                 clustsFile, clusterID, f)
                    raise Exception('clustsFile bad line')
                # should check that TYPE is the same as in clusterID, not bothering for now
            fitWith[clusterID] = fitWithList
        else:
            fitWith[clusterID] = []

        if (gender == 'M') or (gender == 'F'):
            clust2gender[clusterID] = gender
        elif gender != '':
            logger.error("In clustsFile %s for CLUSTER_ID=%s, bad GENDER value: %s", clustsFile, clusterID, gender)
            raise Exception('clustsFile bad line')

        if valid == '0':
            clustIsValid[clusterID] = False
        elif valid == '1':
            clustIsValid[clusterID] = True
        else:
            logger.error("In clustsFile %s for CLUSTER_ID=%s, bad VALID value: %s", clustsFile, clusterID, valid)
            raise Exception('clustsFile bad line')

    clustsFH.close()
    return(clust2samps, samp2clusts, fitWith, clust2gender, clustIsValid)


#############################
# printClustsFile:
# Print data representing clustering results to outFile as a 'clustsFile', see
# parseClustsFile() for details on the data structures (arguments of this function
# and returned by parsClustsFile) and for a spec of the clustsFile format.
#
# Args:
# - clust2samps: dict, key==clusterID, value == list of sampleIDs
# - fitWith: dict, key==clusterID, value == list of clusterIDs
# - clust2gender: dict, key==clusterID (Gonosome clusters only), value == string 'M' or 'F'
# - clustIsValid: dict, key==clusterID, value == Boolean
#
# Returns nothing.
def printClustsFile(clust2samps, fitWith, clust2gender, clustIsValid, outFile):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open clustersFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open clustersFile')

    toPrint = "CLUSTER_ID\tFIT_WITH\tGENDER\tVALID\tSAMPLES\n"
    outFH.write(toPrint)

    # sort clusterIDs - will be number-sorted if numbers were left-padded with zeroes
    for clusterID in sorted(clust2samps.keys()):
        toPrint = clusterID + "\t"
        toPrint += ','.join(sorted(fitWith[clusterID])) + "\t"
        if clusterID in clust2gender:
            toPrint += clust2gender[clusterID] + "\t"
        else:
            toPrint += "\t"
        if clustIsValid[clusterID]:
            toPrint += "1\t"
        else:
            toPrint += "0\t"
        toPrint += ','.join(sorted(clust2samps[clusterID])) + "\n"
        outFH.write(toPrint)
    outFH.close()
