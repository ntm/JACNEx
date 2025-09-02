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
# - dendroFileRoot (str): prefix of filename (including path) for saving dendrograms
#   representing the resulting hierarchical clusterings, along with matching .txt files
#   holding the sampleIDs in dendrogram order
#
# Returns (clust2samps, fitWith, clustIsValid): as defined in clustFile.py parseClustsFile(),
# ie clusterIDs are formatted as TYPE_NUMBER where TYPE is 'A' or 'G', and:
# - clust2samps: dict, key==clusterID, value == list of sampleIDs
# - fitWith: dict, key==clusterID, value == list of clusterIDs
# - clustIsValid: dict, key==clusterID, value == Boolean
def buildClusters(FPMarray, chromType, samples, minSize, dendroFileRoot):
    # data structures to return
    clust2samps = {}
    fitWith = {}
    clustIsValid = {}

    (clusters, fitWithPrev) = clusterize(FPMarray, chromType, samples, minSize, dendroFileRoot, "root")

    # we want clusters to be numbered increasingly by decreasing numbers of samples
    clustIndexes = list(range(len(clusters)))
    clustIndexes.sort(key=lambda ci: len(clusters[ci]), reverse=True)

    # clustIndex2ID[i] is the ID of clusters[i]
    clustIndex2ID = [None] * len(clusters)
    nextClustNb = 1
    for thisClust in clustIndexes:
        # left-pad with leading zeroes if less than 2 digits (for pretty-sorting, won't
        # sort correctly if more than 100 clusters but it's just cosmetic)
        clustID = chromType + '_' + f"{nextClustNb:02}"
        clustIndex2ID[thisClust] = clustID
        nextClustNb += 1
        clust2samps[clustID] = clusters[thisClust]
        clust2samps[clustID].sort()
        fitWith[clustID] = []

    # populate fitWith
    for thisClust in range(len(clusters)):
        if fitWithPrev[thisClust]:
            fitWith[clustIndex2ID[thisClust]] = [clustIndex2ID[thisClust - 1]]

    # define valid clusters, ie size (including valid FIT_WITH) >= minSize , also exclude
    # singletons ie require size (excluding FITWITHs) > 1 (otherwise we can't select
    # CALLable exons in step3)
    clustSizeNoFW = {}

    # need to examine the clusters sorted by number of clusters in their fitWith
    nbFW2clusts = [None] * len(clusters)

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

    return(clust2samps, fitWith, clustIsValid)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################
# Identify clusters of similar samples:
# reduce dimensionality with PCA, perform hierarchical clustering, interpret
# result -> if we have one or two clean clusters we are done, otherwise if the
# data is more complex we recursively apply the same procedure on each child of
# the root node.
# Args are the same as buildClusters() except dendroID: identifies this call in
# the series of recursive clusterize() calls, always starting with "root" and
# then "-L" when going down on left side and "-R" on right.
#
# Return (clusters, fitWithPrev):
# clusters = list of clusters, one cluster is a list of sampleIDs
# fitWithPrev = list of Bools, same length as clusters, True iff cluster[i]
#   is fitWith cluster[i-1]
def clusterize(FPMarray, chromType, samples, minSize, dendroFileRoot, dendroID):
    if (len(samples) < minSize):
        # not enough samples to be valid, no point in doing anything
        return([samples], [False])

    # reduce dimensionality with PCA
    # choosing the optimal number of dimensions is difficult, but should't matter
    # much as long as we're in the right ballpark...
    # In our tests (08/2025) a hard-coded value of 20 works well on all combinations
    # of samples that we tried
    dims = 10
    dims = min(dims, FPMarray.shape[0] - 1, FPMarray.shape[1] - 1)
    pca = sklearn.decomposition.PCA(n_components=dims, svd_solver='full').fit(FPMarray.T)
    # project samples
    samplesInPCAspace = pca.transform(FPMarray.T)

    # hierarchical clustering of the samples projected in the PCA space:
    # - use 'average' method to define the distance between clusters (== UPGMA),
    #   not sure why but AS did a lot of testing and says it is the best choice
    #   when there are different-sized groups;
    # - use 'euclidean' metric to define initial distances between samples;
    # - reorder the linkage matrix so the distance between successive leaves is minimal
    linkageMatrix = scipy.cluster.hierarchy.linkage(samplesInPCAspace, method='average',
                                                    metric='euclidean', optimal_ordering=True)

    # does linkage represent a clean clustering into 1 or 2 clusters?
    (status, si1, si2, c2sFull, fwFull) = interpretLinkage(linkageMatrix, minSize, samples)

    # prepare stuff and call plotDendrogram()
    if chromType == 'A':
        plotFile = dendroFileRoot + "_autosomes_" + dendroID + ".pdf"
    else:
        plotFile = dendroFileRoot + "_gonosomes_" + dendroID + ".pdf"
    if os.path.isfile(plotFile):
        logger.info("pre-existing dendrogram plotFile %s will be squashed", plotFile)
    title = "chromType = " + chromType + " ,  hierarchical clustering ID: " + dendroID
    if status == 0:
        title += " , one clean cluster"
    elif status < 3:
        title += " , two clean clusters"
    else:
        title += " , more than two clusters, will recurse on each child"
    figures.plotDendrograms.plotDendrogram(linkageMatrix, samples, c2sFull, fwFull, title, plotFile)

    # recursively clusterize each child of root if needed, and return
    if status == 0:
        return([samples], [False])
    else:
        # at least 2 clusters, need the lists of sampleIDs under each child
        samples1 = [samples[i] for i in si1]
        samples2 = [samples[i] for i in si2]
        if status == 1:
            return([samples1, samples2], [False, False])
        elif status == 2:
            # second cluster is fitWith  first cluster
            return([samples1, samples2], [False, True])
        else:
            # partition samples and redo clustering
            (clusters1, fitWithPrev1) = clusterize(FPMarray[:, si1], chromType, samples1,
                                                   minSize, dendroFileRoot, dendroID + "-Lo")
            (clusters2, fitWithPrev2) = clusterize(FPMarray[:, si2], chromType, samples2,
                                                   minSize, dendroFileRoot, dendroID + "-Hi")
            clusters1.extend(clusters2)
            fitWithPrev1.extend(fitWithPrev2)
            return(clusters1, fitWithPrev1)


#############################
# Given a linkage matrix, decide if it represents a single cluster, two clusters
# (one for each child of the root node) with one possibly fitWith the other, or
# if we need to redo the clustering independantly on the samples descending
# from each child of the root node.
#
# Args:
# - linkageMatrix: as returned by scipy.cluster.hierarchy.linkage
# - minSize: min number of smaples for a cluster to be valid, used in the heuristic that
#   decides whether children want to merge (smaller clusters have increased desire to merge)
# - sampleIDs (list of strings), same number and order as in linkageMatrix
#
# Returns (status, sampleIndexes1, sampleIndexes2, clust2samps, fitWith):
# (status, sampleIndexes1, sampleIndexes2) are the main beef:
# - if all samples form a single cluster -> (0, None, None)
# - elsif the samples form 2 independant clusters -> (1, si1, si2) where si1 and si2 are the
#     lists of indexes (in samples) of the samples in each cluster
# - elsif the samples form 2 clusters but the second is FitWith the first -> (2, si1, si2)
# - else there are >= 3 clusters, we will need to redo the clustering independantly on
#     the samples descending from each child of the root node -> (3, si1, si2)
# (clust2samps, fitWith) are just for plotting the dendrogram for this specific
# (transient) clustering, the clusterIDs are bogus
def interpretLinkage(linkageMatrix, minSize, sampleIDs):
    numNodes = linkageMatrix.shape[0]
    numSamples = numNodes + 1

    BLzscores = calcBLzscores(linkageMatrix, minSize)

    # merge heuristic: a child wants to merge with its brother if it is non-virtual
    # (ie it has samples and no fitWiths), and:
    # - parent's distance (ie height) is <= startDist, or
    # - parent isn't "too far" relative to the child's intra-cluster branch lengths,
    #   ie BLzscore <= maxZscoreToMerge, or
    # - child is small (< minSize) and child's brother wants to merge.
    # current startDist heurisitic: 10% of highest node
    startDist = linkageMatrix[-1, 2] * 0.1
    maxZscoreToMerge = 3
    logger.debug("in interpretLinkage, using startDist = %.2f", startDist)

    ################
    # a (potential) cluster, identified by its clusterIndex i (0 <= i < numSamples + numNodes), is:
    # - a list of sample indexes, stored in clustSamples[i]
    # - a list of cluster indexes to use as fitWith, in clustFitWith[i]
    clustSamples = [None] * (numSamples + numNodes)
    clustFitWith = [None] * (numSamples + numNodes)

    # When examining internal node i:
    # - if both children want to merge: they are both cleared and node i is created
    # - if only child c1 wants to merge but c2 is "mergeable" (has samples or fitWiths):
    #   c1 will use c2 + clustFitWith[c2] as fitWith, c2 is left as-is, node i is created
    #   with no samples and full fitWith (ie it is mergeable)
    # - if no child wants to merge: they both stay as-is, and node i is created with
    #   no samples and no fitWith (non-mergeable)
    # At the end, the only clusters to create are those with clustSamples != None.

    ################
    # populate clustSamples and clustFitWith

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
            elif clustSamples[children[ci]] and (BLzscores[ni][ci] <= maxZscoreToMerge):
                wantsToMerge[ci] = True
        for ci in range(2):
            if wantsToMerge[1 - ci] and (not wantsToMerge[ci]):
                # ci's brother wants to merge but ci doesn't: if ci is small
                # and non-virtual, it can change its mind
                sizeCi = 1
                if children[ci] >= numSamples:
                    sizeCi = linkageMatrix[children[ci] - numSamples, 3]
                if clustSamples[children[ci]] and (sizeCi < minSize):
                    wantsToMerge[ci] = True

        if wantsToMerge[0] and wantsToMerge[1]:
            clustSamples[thisClust] = clustSamples[c1] + clustSamples[c2]
            clustFitWith[thisClust] = []
            # clear nodes c1 and c2
            clustSamples[c1] = None
            clustSamples[c2] = None
        else:
            # at most one child wants to merge
            # in all cases thisClust will be a virtual no-sample cluster
            clustSamples[thisClust] = None

            # if at least one child is non-mergeable, or if no child wants
            # to merge: don't merge
            if (((clustSamples[c1] is None) and (not clustFitWith[c1])) or
                ((clustSamples[c2] is None) and (not clustFitWith[c2])) or
                ((not wantsToMerge[0]) and (not wantsToMerge[1]))):
                clustFitWith[thisClust] = []

            # else if the only child that wants to merge is already large
            # (2 * minSize), it changes its mind ie it doens't fitWith the other
            elif ((wantsToMerge[0] and (len(clustSamples[c1]) >= 2 * minSize)) or
                  (wantsToMerge[1] and (len(clustSamples[c2]) >= 2 * minSize))):
                clustFitWith[thisClust] = []

            else:
                # exactly one child wants to merge and it's not large and the other is mergeable,
                # thisClust will be mergeable too
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

    ################
    # populate the full clust2samps and fitWith (with bogus clusterIDs), used for plotting
    # the dendrogram of this (transient) clustering
    clust2samps = {}
    fitWith = {}
    # find all non-virtual cluster indexes and build their bogus clusterIDs
    clustIndex2ID = [None] * len(clustSamples)
    nextClustNb = 1
    for thisClust in range(len(clustSamples)):
        if clustSamples[thisClust]:
            clustID = "C" + '_' + f"{nextClustNb:02}"
            clustIndex2ID[thisClust] = clustID
            nextClustNb += 1
            clust2samps[clustID] = [sampleIDs[i] for i in clustSamples[thisClust]]
            clust2samps[clustID].sort()
            fitWith[clustID] = []
    for thisClust in range(len(clustSamples)):
        if clustFitWith[thisClust]:
            clustID = clustIndex2ID[thisClust]
            fitWith[clustID] = [clustIndex2ID[i] for i in clustFitWith[thisClust]]
            fitWith[clustID].sort()

    ################
    if clustSamples[numSamples + numNodes - 1]:
        # we have a single cluster
        retVal = (0, None, None)
    else:
        (c1, c2, dist, size) = linkageMatrix[-1]
        c1 = int(c1)
        c2 = int(c2)
        if clustSamples[c1] and clustSamples[c2]:
            # we have 2 clusters...
            if clustFitWith[c1]:
                # c1 is FitWith c2
                # sanity:
                if (len(clustFitWith[c1]) != 1) or (clustFitWith[c1] != [c2]) or clustFitWith[c2]:
                    logger.error("sanity check failed in interpretLinkage()")
                    raise Exception("interpretLinkage sanity check failed")
                retVal = (2, clustSamples[c2], clustSamples[c1])
            elif clustFitWith[c2]:
                # c2 is FitWith c1
                # sanity:
                if (len(clustFitWith[c2]) != 1) or (clustFitWith[c2] != [c1]) or clustFitWith[c1]:
                    logger.error("sanity check failed in interpretLinkage()")
                    raise Exception("interpretLinkage sanity check failed")
                retVal = (2, clustSamples[c1], clustSamples[c2])
            else:
                # 2 independant clusters
                retVal = (1, clustSamples[c1], clustSamples[c2])
        else:
            # more than 2 clusters, find all samples underneath c1 and c2
            si1 = samplesUnderNode(linkageMatrix, clustSamples, numSamples, c1)
            si2 = samplesUnderNode(linkageMatrix, clustSamples, numSamples, c2)
            retVal = (3, si1, si2)
    retVal += (clust2samps, fitWith)
    return(retVal)


#############################
# return the sorted list of samples present in any cluster underneath "node"
def samplesUnderNode(linkageMatrix, clustSamples, numSamples, node):
    samples = []
    nodesToExamine = [node]
    while (nodesToExamine):
        thisNode = nodesToExamine.pop(0)
        if clustSamples[thisNode]:
            samples.extend(clustSamples[thisNode])
        else:
            # no samples in thisNode, look in its children
            (c1, c2, dist, size) = linkageMatrix[thisNode - numSamples]
            nodesToExamine.extend([int(c1), int(c2)])
    samples.sort()
    return(samples)


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
