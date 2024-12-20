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
import numpy
import os
import scipy.cluster.hierarchy  # hierachical clustering
import sklearn.decomposition  # PCA
import statistics


####### JACNEx modules
import figures.plotDendrograms

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################
# buildClusters:
# find subsets of "similar" samples. Samples within a cluster will be used
# to fit the CN2 (ie wildtype) distribution, which in turn will allow
# to calculate likelihoods of each exon-level CN state.
#
# Args:
# - FPMarray (numpy.ndarray[float]): normalised fragment counts for exons on the
#   chromosome type indicated by chromType, for all samples
# - chromType (string): one of 'A', 'G' indicating that FPMarray holds counts
#   for exons on autosomes or gonosomes
# - samples: list of sampleIDs, same order as the columns of FPMarray
# - minSize: min number of samples (in a cluster + its FIT_WITH friends) to declare
#   the cluster VALID
# - plotFile (str): filename (including path) for saving a dendrogram representing
#   the resulting hierarchical clustering
#
# Returns (clust2samps, fitWith, clustIsValid): as defined in clustFile.py parseClustsFile(),
# ie clusterIDs are formatted as TYPE_NUMBER where TYPE is 'A' or 'G', and:
# - clust2samps: dict, key==clusterID, value == list of sampleIDs
# - fitWith: dict, key==clusterID, value == list of clusterIDs
# - clustIsValid: dict, key==clusterID, value == Boolean
def buildClusters(FPMarray, chromType, samples, minSize, plotFile):
    # reduce dimensionality with PCA
    # choosing the optimal number of dimensions is difficult, but should't matter
    # much as long as we're in the right ballpark -> our heuristic:
    # only keep dimensions that explain > minExplainedVar variance ratio
    minExplainedVar = 0.015

    # we want at most maxDims dimensions (should be more than enough for any dataset)
    maxDims = 20
    maxDims = min(maxDims, FPMarray.shape[0] - 1, FPMarray.shape[1] - 1)
    pca = sklearn.decomposition.PCA(n_components=maxDims, svd_solver='full').fit(FPMarray.T)
    logMess = "in buildClusters while choosing the PCA dimension, explained variance ratio:"
    logger.debug(logMess)
    logMess = numpy.array_str(pca.explained_variance_ratio_, max_line_width=200, precision=3)
    logger.debug(logMess)

    dim = 0
    while ((dim < maxDims) and (pca.explained_variance_ratio_[dim] > minExplainedVar)):
        dim += 1
    if dim == 0:
        logger.warning("in buildClusters: no informative PCA dimensions, very unusual... setting dim=maxDims")
        dim = maxDims
    logger.debug("in buildClusters with chromType=%s, chosen number of PCA dimensions = %i", chromType, dim)
    # project samples and truncate to dim dimensions
    samplesInPCAspace = pca.transform(FPMarray.T)
    samplesInPCAspace = numpy.delete(samplesInPCAspace, numpy.s_[dim:maxDims], 1)

    # hierarchical clustering of the samples projected in the PCA space:
    # - use 'average' method to define the distance between clusters (== UPGMA),
    #   not sure why but AS did a lot of testing and says it is the best choice
    #   when there are different-sized groups;
    # - use 'euclidean' metric to define initial distances between samples;
    # - reorder the linkage matrix so the distance between successive leaves is minimal
    linkageMatrix = scipy.cluster.hierarchy.linkage(samplesInPCAspace, method='average',
                                                    metric='euclidean', optimal_ordering=True)

    # build clusters from the linkage matrix
    (clust2samps, fitWith) = linkage2clusters(linkageMatrix, chromType, samples, minSize)

    # define valid clusters, ie size (including valid FIT_WITH) >= minSize , also require
    # size (excluding FITWITHs) > 1 (otherwise we can't select CALLable exons in step3)
    clustSizeNoFW = {}
    clustIsValid = {}

    # need to examine the clusters sorted by number of (other) clusters in their fitWith
    nbFW2clusts = [None] * len(clust2samps)

    for clust in clust2samps:
        clustSizeNoFW[clust] = len(clust2samps[clust])
        nbFW = len(fitWith[clust])
        if not nbFW2clusts[nbFW]:
            nbFW2clusts[nbFW] = []
        nbFW2clusts[nbFW].append(clust)

    for nbFW in range(len(nbFW2clusts)):
        if nbFW2clusts[nbFW]:
            for clust in nbFW2clusts[nbFW]:
                size = clustSizeNoFW[clust]
                if size == 1:
                    # singleton cluster
                    clustIsValid[clust] = False
                else:
                    for fw in fitWith[clust]:
                        if clustIsValid[fw]:
                            size += clustSizeNoFW[fw]
                    if size >= minSize:
                        clustIsValid[clust] = True
                    else:
                        clustIsValid[clust] = False

    # remove invalid clusters from fitWith
    for clust in clust2samps:
        validFWs = []
        for fw in fitWith[clust]:
            if clustIsValid[fw]:
                validFWs.append(fw)
        fitWith[clust] = validFWs

    # produce and plot dendrogram
    if os.path.isfile(plotFile):
        logger.info("pre-existing dendrogram plotFile %s will be squashed", plotFile)
    title = "chromType = " + chromType + " ,  PCA dimensions = " + str(dim)
    figures.plotDendrograms.plotDendrogram(linkageMatrix, samples, clust2samps, fitWith, clustIsValid, title, plotFile)

    return(clust2samps, fitWith, clustIsValid)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################


#############################
# Build clusters from the linkage matrix.
#
# Args:
# - linkageMatrix: as returned by scipy.cluster.hierarchy.linkage
# - chromType, samples: same args as buildClusters(), used for formatting/populating
#   the returned data structures
# - minSize: min number of smaples for a cluster to be valid, used in the heuristic that
#   decides whether children want to merge (smaller clusters have increased desire to merge)
#
# Returns (clust2samps, fitWith) as specified in the header of buildClusters()
def linkage2clusters(linkageMatrix, chromType, samples, minSize):
    numNodes = linkageMatrix.shape[0]
    numSamples = numNodes + 1

    BLzscores = calcBLzscores(linkageMatrix, minSize)

    # merge heuristic: a child wants to merge with its brother if it doesn't have any
    # fitWith clusters (otherwise the dendrogram can show non-contiguous clusters, this
    # doesn't make sense), and:
    # - parent's distance (ie height) is <= startDist, or
    # - parent isn't "too far" relative to the child's intra-cluster branch lengths,
    #   ie BLzscore <= maxZscoreToMerge, or
    # - child is small (< minSize) and child's brother wants to merge.
    # current startDist heurisitic: 10% of highest node
    startDist = linkageMatrix[-1, 2] * 0.1
    maxZscoreToMerge = 3
    logger.debug("in buildClusters, using startDist = %.2f", startDist)

    ################
    # a (potential) cluster, identified by its clusterIndex i (0 <= i < numSamples + numNodes), is:
    # - a list of sample indexes, stored in clustSamples[i]
    # - a list of cluster indexes to use as fitWith, in clustFitWith[i]
    clustSamples = [None] * (numSamples + numNodes)
    clustFitWith = [None] * (numSamples + numNodes)

    # When examining internal node i:
    # - if both children want to merge: they are both cleared and node i is created
    # - if only child c1 wants to merge: c1 will use c2 + clustFitWith[c2] as fitWith,
    #   c2 is left as-is, node i is created with no samples and full fitWith
    # - if no child wants to merge: they both stay as-is, and node i is created with
    #   no samples and full fitWith
    # At the end, the only clusters to create are those with clustSamples != None.

    ################
    # leaves, ie singleton samples
    for i in range(numSamples):
        clustSamples[i] = [i]
        clustFitWith[i] = []

    # internal nodes
    for ni in range(numNodes):
        (c1, c2, dist, size) = linkageMatrix[ni]
        c1 = int(c1)
        c2 = int(c2)
        children = [c1, c2]
        thisClust = ni + numSamples
        # wantsToMerge: list of 2 Bools, one for each child in the order (c1,c2)
        wantsToMerge = [False, False]
        for ci in range(2):
            if dist <= startDist:
                wantsToMerge[ci] = True
            elif (not clustFitWith[children[ci]]) and (BLzscores[ni][ci] <= maxZscoreToMerge):
                wantsToMerge[ci] = True
        for ci in range(2):
            if wantsToMerge[1 - ci] and (not wantsToMerge[ci]):
                # ci's brother wants to merge but ci doesn't: if ci is small
                # it can change its mind
                sizeCi = 1
                if children[ci] >= numSamples:
                    sizeCi = linkageMatrix[children[ci] - numSamples, 3]
                if (not clustFitWith[children[ci]]) and (sizeCi < minSize):
                    wantsToMerge[ci] = True

        if wantsToMerge[0] and wantsToMerge[1]:
            clustSamples[thisClust] = clustSamples[c1] + clustSamples[c2]
            clustFitWith[thisClust] = []
            # clear nodes c1 and c2
            clustSamples[c1] = None
            clustSamples[c2] = None
        else:
            # at most one child wants to merge
            # in all cases thisClust will be a virtual no-sample cluster (will be useful
            # later if thisClust is used as fitWith for another cluster)
            clustSamples[thisClust] = None
            clustFitWith[thisClust] = clustFitWith[c1] + clustFitWith[c2]
            if clustSamples[c1]:
                clustFitWith[thisClust].append(c1)
            if clustSamples[c2]:
                clustFitWith[thisClust].append(c2)

            if wantsToMerge[0]:
                # c1 wants to merge but not c2
                clustFitWith[c1] += clustFitWith[c2]
                if clustSamples[c2]:
                    # c2 is a real cluster with samples, not a virtual "fitWith" cluster
                    clustFitWith[c1].append(c2)
            elif wantsToMerge[1]:
                clustFitWith[c2] += clustFitWith[c1]
                if clustSamples[c1]:
                    clustFitWith[c2].append(c1)
            # else c1 and c2 don't change

    ################
    # populate the clustering data structures from the Tmp lists, with proper formatting
    clust2samps = {}
    fitWith = {}

    # populate clustIndex2ID with correctly constructed clusterIDs for each non-virtual cluster index
    clustIndex2ID = [None] * len(clustSamples)

    # we want clusters to be numbered increasingly by decreasing numbers of samples
    clustIndexes = []
    # find all non-virtual cluster indexes
    for thisClust in range(len(clustSamples)):
        if clustSamples[thisClust]:
            clustIndexes.append(thisClust)
    # sort them by decreasing numbers of samples
    clustIndexes.sort(key=lambda ci: len(clustSamples[ci]), reverse=True)

    nextClustNb = 1
    for thisClust in clustIndexes:
        # left-pad with leading zeroes if less than 2 digits (for pretty-sorting, won't
        # sort correctly if more than 100 clusters but it's just cosmetic)
        clustID = chromType + '_' + f"{nextClustNb:02}"
        clustIndex2ID[thisClust] = clustID
        nextClustNb += 1
        clust2samps[clustID] = [samples[i] for i in clustSamples[thisClust]]
        clust2samps[clustID].sort()
        fitWith[clustID] = []

    for thisClust in clustIndexes:
        if clustFitWith[thisClust]:
            clustID = clustIndex2ID[thisClust]
            fitWith[clustID] = [clustIndex2ID[i] for i in clustFitWith[thisClust]]
            fitWith[clustID].sort()

    return (clust2samps, fitWith)


#############################
# Given a hierarchical clustering (linkage matrix), calculate branch-length pseudo-Zscores:
# for each internal node ni (row index in linkageMatrix), BLzscores[ni] is a list of 2
# floats, one for each child cj (j==0 or 1, same order as they appear in linkageMatrix):
# BLzscores[ni][j] = [d(ni, cj) - W * mean(BLs[cj])] / (stddev(BLs[cj]) + 1)
# [convention: BLzscores[clusti][cj] == 0 if cj is a leaf]
# W == max(minSize / size(cj) , 1) is a weight that decreases the BLzscore for small children,
# ie children smaller than minSize have inflated desire to merge with their brother.
# Note: divisor is stddev+1 to avoid divByZero and/or inflated zscores when BLs are equal
# or very close.
#
# Args:
# - linkageMatrix: as returned by scipy.cluster.hierarchy.linkage
# - minSize: min number of samples for a cluster to be valid, used for the weight
#
# Returns BLzscores, a list (size == number of rows in linkageMatrix) of lists of 2 floats,
# as defined above.
def calcBLzscores(linkageMatrix, minSize):
    numNodes = linkageMatrix.shape[0]
    numSamples = numNodes + 1
    BLzscores = [None] * numNodes
    # for each internal node index ni, BLs[ni] will be the list of branch lengths
    # (floats) below the node
    BLs = [None] * numNodes
    for ni in range(numNodes):
        (c1, c2, dist, size) = linkageMatrix[ni]
        c1 = int(c1)
        c2 = int(c2)
        BLzscores[ni] = []
        BLs[ni] = []
        for child in (c1, c2):
            if child < numSamples:
                # leaf
                BLzscores[ni].append(0)
                BLs[ni].append(dist)
            else:
                child -= numSamples
                thisBL = dist - linkageMatrix[child, 2]
                childBLs = BLs[child]
                weight = max(minSize / linkageMatrix[child, 3], 1)
                thisZscore = (thisBL - weight * statistics.mean(childBLs)) / (statistics.stdev(childBLs) + 1)
                BLzscores[ni].append(thisZscore)
                BLs[ni].append(thisBL)
                BLs[ni].extend(childBLs)
    return(BLzscores)
