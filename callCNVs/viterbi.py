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


import concurrent.futures
import logging
import math
import multiprocessing
import numpy

####### JACNEx modules
import callCNVs.transitions

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

######################################
# viterbiAllSamples:
# given a fully specified HMM, Call CNVs with viterbiOneSample() in parallel for
# each sample in samples (==nbSamples in likelihoods).
# See viterbiOneSample() top-of-function comments for methodology info.
#
# Args:
# - likelihoods: numpy 3D-array of floats of size nbSamples * nbExons * nbStates,
#   likelihoods[s,e,cn] is the likehood of state cn for exon e in sample s
#   (NOCALL exons must have all likelihoods == -1)
# - Ecodes: exon filtering codes
# - samples: list of nbSamples sampleIDs (==strings)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - transMatrix (ndarray[floats] dim nbStates*nbStates): base transition probas between states
# - priors (ndarray dim nbStates): prior probabilities for each state
# - dmax [int]: param for adjustTransMatrix()
# - minGQ [float]: minimum Genotype Quality (GQ)
# - jobs (int): Number of samples to process in parallel.
#
# Returns a list of CNVs, a CNV is a list (types [int, int, int, float, int]):
# [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleIndex]
# where firstExonIndex and lastExonIndex are indexes in the provided exons,
# and sampleIndex is the index in the provided samples
def viterbiAllSamples(likelihoods, Ecodes, samples, exons, transMatrix, priors, dmax, minGQ, jobs):
    CNVs = []

    # sanity checks
    if len(samples) != likelihoods.shape[0]:
        logger.error("SANITY: numbers of samples and of rows in likelihoods inconsistent")
        raise Exception("viterbiAllSamples sanity-check failed")
    if len(exons) != likelihoods.shape[1]:
        logger.error("SANITY: numbers of exons and of cols in likelihoods inconsistent")
        raise Exception("viterbiAllSamples sanity-check failed")
    if len(transMatrix) != likelihoods.shape[2]:
        logger.error("SANITY: numbers of states in transMatrix and in likelihoods inconsistent")
        raise Exception("viterbiAllSamples sanity-check failed")
    if len(transMatrix) != len(priors):
        logger.error("SANITY: numbers of states in transMatrix and in priors inconsistent")
        raise Exception("viterbiAllSamples sanity-check failed")

    ##################
    # Define nested callback for processing viterbiOneSample() result (so CNVs is in scope).
    # sampleDone:
    # arg: a Future object returned by ProcessPoolExecutor.submit(viterbiOneSample).
    # If something went wrong, log and propagate exception, otherwise store CNVs.
    def sampleDone(futureRes):
        e = futureRes.exception()
        if e is not None:
            logger.error("viterbiOneSample() failed for sample %s", str(e))
            raise(e)
        else:
            CNVs.extend(futureRes.result())

    ##################
    with concurrent.futures.ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('fork')) as pool:
        for si in range(len(samples)):
            futureRes = pool.submit(viterbiOneSample, likelihoods[si, :, :], Ecodes, si,
                                    samples[si], exons, transMatrix, priors, dmax, minGQ)
            futureRes.add_done_callback(sampleDone)

    return(CNVs)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

######################################
# viterbiOneSample:
# given a fully specified HMM, this function calls CNVs for one sample: it
# implements the Viterbi algorithm to find the most likely sequence of copy-number
# states that explain the observations (likelihoods).
#
# The underlying HMM is defined by:
# - one state per copy number (0==homodel, 1==heterodel, 2==WT, 3==CN3+==DUP)
# - prior propabilities for each state;
# - transition probabilities that depend on the distance to the next exon - they begin when
#   distance=0 at the "base" values defined in transMatrix, and are smoothly adjusted following
#   a power law until they reach the prior probabilities at dist dmax;
# - emission likelihoods of the sample's FPM in each state and for each exon.
#
# NOTE: Ecodes==1 ie CALLED-CN1-RESTRICTED exons can be in any state (including CN1),
# but they cannot start or end a CN1 event, only extend it
#
# Args:
# - likelihoods (ndarray[floats] dim nbExons*nbStates): emission likelihoods of
#   each state for each exon for one sample.
# - Ecodes: exon filtering codes
# - sampleIndex [int] (for returned CNVs)
# - sampleID [str] (for log messages)
# - exons [list of lists[str, int, int, str]]: exon infos [chr, START, END, EXONID].
# - transMatrix (ndarray[floats] dim nbStates*nbStates): base transition probas between states
# - priors (ndarray dim nbStates): prior probabilities for each state
# - dmax [int]: param for adjustTransMatrix()
# - minGQ [float]: minimum Genotype Quality (GQ)
#
# Returns:
# - CNVs (list of list[int, int, int, float, int]): list of called CNVs,
#   a CNV is a list [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleIndex].
def viterbiOneSample(likelihoods, Ecodes, sampleIndex, sampleID, exons, transMatrix, priors, dmax, minGQ):
    try:
        CNVs = []
        nbStates = len(transMatrix)

        # Step 1: Initialize variables
        # probsPrev[s]: probability of the most likely path ending in state s at previous exon,
        # initialize path root at CN2
        probsPrev = numpy.zeros(nbStates, dtype=numpy.float128)
        probsPrev[2] = 1
        # chrom and end of previous exon - init at -dmax so first exon uses the priors
        prevChrom = exons[0][0]
        prevEnd = -dmax
        # Ecode of prev exon (0==CALLED or 1==CALLED-CN1-RESTRICTED, no-calls are skipped)
        prevEcode = 0

        # temp data structures used by buildCNVs() and reset whenever it is called,
        # see buildCNVs() spec for info
        (calledExons, path, bestPathProbas, CN2FromCN2Probas) = ([], [], [], [])

        # Step 2: viterbi forward algorithm
        for exonIndex in range(len(exons)):
            if likelihoods[exonIndex, 0] == -1:
                # exon is no-call => skip
                continue

            if exons[exonIndex][0] != prevChrom:
                # changing chroms:
                appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas)
                CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleIndex, minGQ))
                # reinit with CN2 as path root
                probsPrev[:] = 0
                probsPrev[2] = 1
                prevChrom = exons[exonIndex][0]
                prevEnd = -dmax
                prevEcode = 0
                (calledExons, path, bestPathProbas, CN2FromCN2Probas) = ([], [], [], [])

            # accumulators with current exon data for populating the buildCNVs() structures
            # bestPrevState defaults to CN2
            bestPrevState = numpy.full(nbStates, 2, dtype=numpy.int8)
            # probsCurrent[s]: probability of the most likely path ending in state s at the current exon
            probsCurrent = numpy.zeros(nbStates, dtype=numpy.float128)

            # adjust transition probabilities
            distFromPrevEx = exons[exonIndex][1] - prevEnd - 1
            adjustedTransMatrix = callCNVs.transitions.adjustTransMatrix(transMatrix, priors, distFromPrevEx, dmax)

            # calculate proba of the most likely paths ending in each state for current exon
            for currentState in range(nbStates):
                probMax = 0
                prevStateMax = 2
                if (Ecodes[exonIndex] == 1) and (currentState == 1):
                    # current exon is CALLED-CN1-RESTRICTED: it can only be CN1 if extending a HET-DEL,
                    # it cannot start a new one
                    probsCurrent[1] = probsPrev[1] * adjustedTransMatrix[1, 1] * likelihoods[exonIndex, 1]
                    bestPrevState[1] = 1
                else:
                    for prevState in range(nbStates):
                        if (prevEcode == 1) and (prevState == 1) and (currentState != 1):
                            # previous exon was CALLED-CN1-RESTRICTED: its CN1 best path (ie HET-DEL) can only
                            # be extended as a longer HET-DEL, ie the path can only go to CN1 in the current exon
                            continue
                        # probability of path coming from prevState to currentState
                        prob = (probsPrev[prevState] *
                                adjustedTransMatrix[prevState, currentState] *
                                likelihoods[exonIndex, currentState])
                        if prob > probMax:
                            probMax = prob
                            prevStateMax = prevState
                    probsCurrent[currentState] = probMax
                    bestPrevState[currentState] = prevStateMax

            # if all best paths for current exon have zero probability (ie they all underflowed)
            if not probsCurrent.any():
                thisEx = exons[exonIndex]
                logger.warning("for sample %s, all best paths to exon %s:%i-%i underflowed to zero proba.",
                               sampleID, thisEx[0], thisEx[1], thisEx[2])
                logger.warning("This should be very rare, if not please report it.")
                # we'll try to build CNVs from whichever state was most likely in the last exon
                appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas)
                # make sure we will backtrack-and-reset (should already be all-2s, but whatever)
                bestPrevState[:] = 2

            # if all states at currentExon have the same predecessor state and that state is CN2:
            # backtrack from [previous exon, CN2] and reset
            if numpy.all(bestPrevState == 2):
                CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleIndex, minGQ))
                # reinit with CN2 as path root in prev exon
                probsCurrent[:] = adjustedTransMatrix[2, :] * likelihoods[exonIndex, :]
                # bestPrevState is already all-2's
                (calledExons, path, bestPathProbas, CN2FromCN2Probas) = ([], [], [], [])

            # OK, update all structures and move to next exon
            numpy.copyto(probsPrev, probsCurrent)
            prevEnd = exons[exonIndex][2]
            prevEcode = Ecodes[exonIndex]
            calledExons.append(exonIndex)
            path.append(bestPrevState)
            bestPathProbas.append(probsCurrent)
            CN2FromCN2Probas.append(adjustedTransMatrix[2, 2] * likelihoods[exonIndex, 2])

        # Final CNVs for the last exons
        appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas)
        CNVs.extend(buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleIndex, minGQ))

        return(CNVs)

    except Exception as e:
        logger.error("failed for sample %s in exon %i: %s", sampleID, exonIndex, str(e))
        raise Exception(sampleID)


######################################
# appendBogusExon:
# append a bogus exon if CN2 isn't the last calledExon's most likely state.
# This bogus exon is built so that the best path ending in its CN2 state is
# almost identical to the best path ending in calledExon[-1]'s most likely state
# (ie same path except for the appended CN2, same bestPathProbas, same score).
def appendBogusCN2Exon(calledExons, path, bestPathProbas, CN2FromCN2Probas):
    lastState = bestPathProbas[-1].argmax()
    if lastState != 2:
        calledExons.append(-1)
        path.append(numpy.array([0, 0, lastState, 0], dtype=numpy.int8))
        bestPathProbas.append(numpy.array([0, 0, bestPathProbas[-1][lastState], 0], dtype=numpy.float128))
        CN2FromCN2Probas.append(1.0)


######################################
# buildCNVs
# Identify CNVs (= consecutive exons with the same CN) in the most-likely path ending in
# state CN2 in the last calledExon, and calculate the associated "qualityScore" (see below).
# Requirements: the called exon preceding calledExons[0] (called the "path root") must be in
# state CN2 in the most likely path
# NOTE: the best-path-ends-in-CN2 precondition means that in the NEXT exon, all best paths
# came from CN2 in calledExons[-1]; but it's perfectly possible that path[-1].argmax() != 2
#
# Args:
# - calledExons [list of ints]: list of called exonIndexes to process here
# - path (list of len(calledExons) ndarrays of nbStates ints):
#   path[e][s] == state of called exon preceding calledExons[e] that produces the max
#   proba for state s at exon calledExons[e]
# - bestPathProbas (list of len(calledExons) ndarrays of nbStates floats):
#   bestPathProbas[e][s] == proba of most likely path ending in state s at exon
#   calledExons[e] and starting at the path root
# - CN2FromCN2Probas (list of len(calledExons) floats): CN2FromCN2Probas[e] == proba of
#   the transition to CN2 in calledExons[e] from CN2 in previous exon
# - sampleIndex [int]
# - minGQ [float]: minimum Genotype Quality (GQ)
#
# Returns a list of CNVs, a CNV == [CNType, startExon, endExon, qualityScore, sampleIndex]:
# - CNType is 0-3 (== CN)
# - startExon and endExon are indexes (in the global exons list) of the first and
#   last exons defining this CNV
# - qualityScore = log10 of ratio between the proba of most likely path between the called
#   exons immediately preceding and immediately following the CNV, and the proba of
#   the CN2-only path between the same exons, capped here at maxQualityScore
def buildCNVs(calledExons, path, bestPathProbas, CN2FromCN2Probas, sampleIndex, minGQ):
    # max quality score of the returned CNVs, hard-coded here - should be >= maxGQ
    # hard-coded in printCallsFile()
    maxQualityScore = 100
    CNVs = []

    # if each exon's bestPath-to-CN2 comes from CN2 in prev exon,
    # the best path is necessarily all-CN2 => there's nothing to build
    # (NOTE this test is also true if calledExons is empty)
    if all((p[2] == 2) for p in path):
        return(CNVs)

    # build ndarray of states that form the most likely path, must start from the end
    mostLikelyStates = numpy.zeros(len(calledExons), dtype=numpy.int8)
    mostLikelyStates[-1] = 2
    currentState = 2
    for cei in range(len(calledExons) - 1, 0, -1):
        currentState = path[cei][currentState]
        mostLikelyStates[cei - 1] = currentState

    # sanity: path root == CN2
    if path[0][currentState] != 2:
        logger.error("in buildCNVs(), sanity check failed: path root != CN2")
        raise Exception(sampleIndex)

    # now walk through the path of most likely states, constructing CNVs as we go
    firstExonInCurrentState = 0
    currentState = mostLikelyStates[0]
    CN2PathProba = CN2FromCN2Probas[0]

    for cei in range(1, len(calledExons)):
        if mostLikelyStates[cei] == currentState:
            # next exon is in same state, just update CN2PathProba
            CN2PathProba *= CN2FromCN2Probas[cei]
        else:
            if (currentState != 2):
                # we are changing states and current wasn't CN2, create CNV
                # score = log10 of ratio between best path proba and CN2-only path proba,
                # use maxQualityScore if CN2-only path proba underflows to zero
                qualityScore = maxQualityScore
                CN2PathProba *= CN2FromCN2Probas[cei]
                if (CN2PathProba > 0):
                    qualityScore = bestPathProbas[cei][mostLikelyStates[cei]] / CN2PathProba
                    if firstExonInCurrentState > 0:
                        # we want the proba of the path starting at the exon immediately
                        # preceding the CNV, not starting at the path root
                        qualityScore /= bestPathProbas[firstExonInCurrentState - 1][mostLikelyStates[firstExonInCurrentState - 1]]
                    qualityScore = math.log10(qualityScore)
                    qualityScore = min(qualityScore, maxQualityScore)

                if qualityScore >= minGQ:
                    CNVs.append([currentState, calledExons[firstExonInCurrentState],
                                 calledExons[cei - 1], qualityScore, sampleIndex])
            # in any case we changed states, update accumulators
            firstExonInCurrentState = cei
            currentState = mostLikelyStates[cei]
            CN2PathProba = CN2FromCN2Probas[cei]

    return(CNVs)
