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


import concurrent.futures
import datetime
import logging
import math
import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.figure
import numpy
import os

####### JACNEx modules
import callCNVs.likelihoods
import countFrags.bed

# prevent matplotlib and PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# For each sample with at least one called CNV, produce a PDF file in plotDir with
# one plot per exon covered (+ the called exons immediately surrounding) each CNV.
# We process jobs samples in parallel.
#
# Args:
# - CNVs: list of CNVs for one cluster, as returned by viterbiAllSamples()
# - sampleIDs: list of nbSOIs sampleIDs (==strings), must be in the same order
#   as the corresponding samplesOfInterest columns in exonFPMs
# - exons (list[str, int, int, str]): exon information.
# - Ecodes (numpy.ndarray[ints]): exons filtering codes.
# - exonFPMs: 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is the FPM
#   count for exon e in sample s (includes samples in FITWITH clusters)
# - samplesOfInterest: 1D-array of bools of size nbSamples, value==True iff the sample
#   is in the cluster of interest (vs being in a FITWITH cluster)
# - isHaploid (bool): Whether the samples are haploid in this cluster.
# - CN0sigma (float): Scale parameter for CN0 distribution.
# - CN2means (numpy.ndarray[floats]): means for CN2 distribution.
# - CN2sigmas (numpy.ndarray[floats]): standard deviations for CN2 distribution.
# - clusterID [str]
# - plotDir [str]: subdir where PDFs are produced, pre-existing files are squashed.
# - jobs [int]: number of samples to process in parallel
#
# Produces plotFile (pdf format), returns nothing.
def plotCNVs(CNVs, sampleIDs, exons, Ecodes, exonFPMs, samplesOfInterest,
             isHaploid, CN0sigma, CN2means, CN2sigmas, clusterID, plotDir, jobs):
    # early return if no CNVs
    if len(CNVs) == 0:
        return()

    matplotlib.use('pdf')
    # sort CNVs by sampleIndex-chrom-start
    CNVs.sort(key=lambda CNV: (CNV[4], CNV[1]))
    # append bogus final CNV (won't be processed, just needs impossible sampleIndex)
    CNVs.append([0, 0, 0, 0, -1])

    ##################
    # Define nested callback for propagating exceptions from children if any
    def sampleDone(futureRes):
        e = futureRes.exception()
        if e is not None:
            logger.error("plotCNVsOneSample() failed for a sample: %s", str(e))
            raise(e)
    ##################

    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        sampleIndex = CNVs[0][4]
        CNVsOneSample = []
        for CNV in CNVs:
            if (CNV[4] == sampleIndex):
                CNVsOneSample.append(CNV)
            else:
                # changing samples
                futureRes = pool.submit(plotCNVsOneSample, CNVsOneSample, sampleIDs[sampleIndex],
                                        exons, Ecodes, exonFPMs, samplesOfInterest, isHaploid,
                                        CN0sigma, CN2means, CN2sigmas, clusterID, plotDir)
                futureRes.add_done_callback(sampleDone)
                sampleIndex = CNV[4]
                CNVsOneSample = [CNV]


####################################################
# checkRegionsToPlot:
# do basic syntactic sanity-check of regionsToPlot, which should be a comma-separated
# list of sampleID:chr:start-end .
# If AOK, return a list of lists [str, str, int, int] holding [sampleID, chrom, start, end];
# else raise exception.
def checkRegionsToPlot(regionsToPlot):
    regions = []
    RTPs = regionsToPlot.split(',')
    for rtp in RTPs:
        rtpList = rtp.split(':')
        if len(rtpList) != 3:
            raise Exception("badly formatted regionToPlot, need 3 ':'-separated fields: " + rtp)
        startEnd = rtpList[2].split('-')
        if len(startEnd) != 2:
            raise Exception("badly formatted regionToPlot, need coords as start-end: " + rtp)
        (start, end) = startEnd
        try:
            start = int(start)
            end = int(end)
            if (start < 0) or (start > end):
                raise Exception()
        except Exception:
            raise Exception("badly formatted regionToPlot, must have 0 <= start <= end: " + rtp)
        regions.append([rtpList[0], rtpList[1], start, end])
    return(regions)


###################
# validate and pre-process each regionsToPlot:
# - does the sampleID exist? In what clusters?
# - does the chrom exist? In auto or gono?
# - are there any exons in the coords?
# If NO to any, log the issue and ignore this regionToPlot;
# if YES to all, populate and return clust2regions:
# key==clusterID, value==Dict with key==sampleID and value==list of exonIndexes
# (in the cluster's exons, auto or gono)
def preprocessRegionsToPlot(regionsToPlot, autosomeExons, gonosomeExons, samp2clusts, clustIsValid):
    clust2regions = {}
    if regionsToPlot == "":
        return(clust2regions)

    autosomeExonNCLs = countFrags.bed.buildExonNCLs(autosomeExons)
    gonosomeExonNCLs = countFrags.bed.buildExonNCLs(gonosomeExons)
    for region in checkRegionsToPlot(regionsToPlot):
        (sampleID, chrom, start, end) = region
        regionStr = sampleID + ':' + chrom + ':' + str(start) + '-' + str(end)
        if sampleID not in samp2clusts:
            logger.warning("ignoring bad regionToPlot %s, sample doesn't exist", regionStr)
            continue
        if chrom in autosomeExonNCLs:
            clustType = 'A_'
            exonNCLs = autosomeExonNCLs
        elif chrom in gonosomeExonNCLs:
            clustType = 'G_'
            exonNCLs = gonosomeExonNCLs
        else:
            logger.warning("ignoring bad regionToPlot %s, chrom doesn't exist", regionStr)
            continue

        clusterID = ""
        for clust in samp2clusts[sampleID]:
            if clust.startswith(clustType):
                clusterID = clust
                break

        if not clustIsValid[clusterID]:
            logger.warning("ignoring regionToPlot %s, sample belongs to invalid cluster %s",
                           regionStr, clusterID)
            continue

        overlappedExons = exonNCLs[chrom].find_overlap(start, end)
        if not overlappedExons:
            logger.warning("ignoring regionToPlot %s, region doesn't overlap any exons", regionStr)
            continue
        if clusterID not in clust2regions:
            clust2regions[clusterID] = {}
        if sampleID not in clust2regions[clusterID]:
            clust2regions[clusterID][sampleID] = []
        for exon in overlappedExons:
            exonIndex = exon[2]
            clust2regions[clusterID][sampleID].append(exonIndex)

    return(clust2regions)


###################################
# This function generates a PDF file with histograms of FPM values and overlays
# model likelihoods for different copy number states (CN0, CN1, CN2, CN3).
# Args:
# - exons (list[str, int, int, str]): exon information.
# - exonsToPlot (dict): key==exonIndex, value==list of lists[sampleIndex, sampleID] for
#                       which we need to plot the FPMs and CN0-CN3+ models
# - Ecodes (numpy.ndarray[ints]): exons filtering codes.
# - exonFPMs: 2D-array of floats of size nbExons * nbSamples, FPMs[e,s] is the FPM
#   count for exon e in sample s (includes samples in FITWITH clusters, these are
#   used for fitting the CN2)
# - samplesOfInterest: 1D-array of bools of size nbSamples, value==True iff the sample
#   is in the cluster of interest (vs being in a FITWITH cluster)
# - isHaploid (bool): Whether the sample is haploid.
# - CN0sigma (float): Scale parameter for CN0 distribution.
# - CN2means (numpy.ndarray[floats]): means for CN2 distribution.
# - CN2sigmas (numpy.ndarray[floats]): standard deviations for CN2 distribution.
# - clusterID [str]
# - plotDir [str]: Folder path to save the generated PDF.
# Produces plotFile (pdf format), returns nothing.
def plotQCExons(exons, exonsToPlot, Ecodes, exonFPMs, samplesOfInterest, isHaploid, CN0sigma, CN2means, CN2sigmas, clusterID, plotDir):
    # return immediately if exonsToPlot is empty
    if not exonsToPlot:
        return

    matplotlib.use('pdf')
    # construct the filename
    current_time = datetime.datetime.now().strftime("%y%m%d_%H-%M-%S")
    plotFile = os.path.join(plotDir, f'{clusterID}_plotExons_{current_time}.pdf')
    matplotFile = matplotlib.backends.backend_pdf.PdfPages(plotFile)

    for thisExon in exonsToPlot.keys():
        (thisSample, sampleID) = exonsToPlot[thisExon]
        plotExon(thisSample, sampleID, thisExon, exons, Ecodes, exonFPMs, samplesOfInterest,
                 isHaploid, CN0sigma, CN2means, CN2sigmas, False, clusterID, matplotFile)
    matplotFile.close()


###############################################################################
############################ PRIVATE FUNCTIONS #################################
###############################################################################
# Return a list of 4 strings: the legend labels for each CN model, one string per
# CN state, labels[1]=="" if isHaploid
def getLabels(isHaploid, CN0Sigma, CN2Mean, CN2Sigma):
    # parameters of CN1 and CN3 (MUST MATCH the models in likelihooods.py)
    CN1Mean = CN2Mean / 2
    CN1Sigma = CN2Sigma / 2
    # CN3Sigma = 0.5    constant => not displayed
    CN3Loc = CN2Mean + 2 * CN2Sigma
    CN3Mu = math.log(CN2Mean)
    if isHaploid:
        CN3Mu += math.log(2)

    if isHaploid:
        labels = [fr'CN0 ($\sigma$={CN0Sigma:.2f})',
                  "",
                  fr'CN1 ($\mu$={CN2Mean:.2f}, $\sigma$={CN2Sigma:.2f})',
                  fr'CN2+ ($\mu$={CN3Mu:.2f}, $\text{{loc}}$={CN3Loc:.2f})']
    else:
        labels = [fr'CN0 ($\sigma$={CN0Sigma:.2f})',
                  fr'CN1 ($\mu$={CN1Mean:.2f}, $\sigma$={CN1Sigma:.2f})',
                  fr'CN2 ($\mu$={CN2Mean:.2f}, $\sigma$={CN2Sigma:.2f})',
                  fr'CN3+ ($\mu$={CN3Mu:.2f}, $\text{{loc}}$={CN3Loc:.2f})']

    return labels


#####################
# For one sample+exon, plot histogram of FPM values and overlays model likelihoods
# for each copy number state (CN0, CN1, CN2, CN3).
# Save as a fig in matplotFile.
# Args: most are the same as plotCNVs() args, special notes:
# thisSample: index of sample among samplesOfInterest (ie not counting FITWITH samples)
# CNV: for title, ignored if False (eg if plotting uncalled exons for QC)
def plotExon(thisSample, sampleID, thisExon, exons, Ecodes, exonFPMs, samplesOfInterest,
             isHaploid, CN0sigma, CN2means, CN2sigmas, CNV, clusterID, matplotFile):
    CNcolors = ['red', 'orange', 'green', 'purple']
    ECodeSTR = {0: 'CALLED', 1: 'CALLED-WITHOUT-CN1', -1: 'NOCALL:NOT-CAPTURED',
                -2: 'NOCALL:FIT-CN2-FAILED', -3: 'NOCALL:CN2-LOW-SUPPORT', -4: 'NOCALL:CN0-TOO-CLOSE'}

    # don't plot NOT-CAPTURED exons, there's nothing interesting to plot
    # (also ignore FIT-CN2-FAILED exons but I've never seen one)
    if (Ecodes[thisExon] == -1) or (Ecodes[thisExon] == -2):
        logging.debug((f"Sample {sampleID}, not plotting exon {exons[thisExon][0]}:"
                       f"{exons[thisExon][1]}-{exons[thisExon][2]} {exons[thisExon][3]}:"
                       f"{ECodeSTR[Ecodes[thisExon]]}"))
        return()

    # number of bins for the histograms: try to use numSamples/2 (including FITWITH samples),
    # but at least 40 and at most 100
    nbBins = int(min(max(40, exonFPMs.shape[1] / 2), 100))

    fpms = exonFPMs[thisExon, :]
    # on x axis: plot up to 20% beyond max FPM
    fpmMax = max(fpms) * 1.2
    # 1000 evenly-spaced x coordinates, for smooth plots
    xcoords = numpy.linspace(0, fpmMax, 1000)

    # as args of calcLikelihoods we need numpy arrays with a single item, use a slice so
    # we get a view (rather than making a new array with eg numpy.array([Ecodes[thisExon]]))
    pdfs = callCNVs.likelihoods.calcLikelihoods(
        xcoords.reshape(1, -1), CN0sigma, Ecodes[thisExon:thisExon + 1],
        CN2means[thisExon:thisExon + 1], CN2sigmas[thisExon:thisExon + 1], isHaploid, True)

    labels = getLabels(isHaploid, CN0sigma, CN2means[thisExon], CN2sigmas[thisExon])

    fpmSOI = fpms[samplesOfInterest]
    fpmNonSOI = fpms[~samplesOfInterest]

    if isHaploid:
        limY = 2 * max(pdfs[:, :, 2])
    else:
        limY = 1.2 * max(pdfs[:, :, 1])

    fig = matplotlib.figure.Figure(figsize=(8, 6))
    ax = fig.subplots()

    yhist = ax.hist(fpmSOI,
                    bins=nbBins,
                    range=(0, xcoords[-1]),
                    label='FPMs (this cluster)',
                    density=True,
                    color='lightblue',
                    edgecolor='black')[0]
    limY = max(limY, 1.2 * max(yhist))

    # histogram of FITWITH samples: on top, but with transparency
    if len(fpmNonSOI) != 0:
        ax.hist(fpmNonSOI,
                bins=nbBins,
                range=(0, xcoords[-1]),
                label='FPMs (FITWITH cluster(s))',
                density=True,
                color='grey',
                edgecolor='black',
                alpha=0.5)

    # plot CN models
    for cnState in range(4):
        if (labels[cnState] != ''):
            if (Ecodes[thisExon] < 0) or ((Ecodes[thisExon] == 1) and (cnState == 1)):
                # dashed lines for all CNs of NOCALL exons and for CN1 of CALL-WITHOUT-CN1 exons
                linestyleFIT = 'dashed'
            else:
                linestyleFIT = 'solid'
            ax.plot(xcoords,
                    pdfs[:, :, cnState].reshape(-1),
                    linewidth=3,
                    linestyle=linestyleFIT,
                    color=CNcolors[cnState],
                    label=labels[cnState])

    # vertical dashed line for thisSample
    ax.axvline(fpmSOI[thisSample],
               color='blue',
               linewidth=2,
               linestyle='dashed',
               label=f'{sampleID} ({fpmSOI[thisSample]:.2f} FPM)')

    title = f"{sampleID} in cluster {clusterID}\n"
    if (CNV):
        (cn, startExi, endExi, qualScore, sampleIndex) = CNV
        if isHaploid:
            if cn == 0:
                title += "HEMI-DEL "
            elif cn == 3:
                title += "DUP "
            else:
                raise Exception("plotExon sanity: CNV isHaploid but CN != 0 or 3, impossible")
        else:
            if cn == 0:
                title += "HOMO-DEL "
            elif cn == 1:
                title += "HET-DEL "
            else:
                title += "DUP "
        title += f"{exons[startExi][0]}:{exons[startExi][1]}-{exons[endExi][2]}\n"
    title += f"{ECodeSTR[Ecodes[thisExon]]} exon {exons[thisExon][0]}:{exons[thisExon][1]}-{exons[thisExon][2]}"
    if (CNV):
        if (thisExon < CNV[1]):
            title += " (before CNV)"
        elif (thisExon <= CNV[2]):
            title += " (in CNV)"
        else:
            title += " (after CNV)"
    ax.set_title(title)
    ax.set_xlabel("FPM")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, xcoords[-1])
    ax.set_ylim(0, limY)
    matplotFile.savefig(fig)


#####################
# Given a CNV called in a sample, plot all exons from the CALLed exon preceding the CNV
# to the CALLed exon following the CNV. Save to matplotFile.
def plotExonsOneCNV(CNV, sampleID, exons, Ecodes, exonFPMs, samplesOfInterest,
                    isHaploid, CN0sigma, CN2means, CN2sigmas, clusterID, matplotFile):
    # find CALLed exons surrounding the CNV
    (cn, startExi, endExi, qualScore, sampleIndex) = CNV
    chrom = exons[startExi][0]
    exonBefore = startExi
    if (exonBefore > 0) and (exons[exonBefore - 1][0] == chrom):
        exonBefore -= 1
    while ((exonBefore > 0) and (Ecodes[exonBefore] < 0) and (exons[exonBefore - 1][0] == chrom)):
        exonBefore -= 1
    exonAfter = endExi
    if (exonAfter + 1 < len(exons)) and (exons[exonAfter + 1][0] == chrom):
        exonAfter += 1
    while ((exonAfter + 1 < len(exons)) and (Ecodes[exonAfter] < 0) and (exons[exonAfter + 1][0] == chrom)):
        exonAfter += 1

    for thisExon in range(exonBefore, exonAfter + 1):
        plotExon(sampleIndex, sampleID, thisExon, exons, Ecodes, exonFPMs, samplesOfInterest,
                 isHaploid, CN0sigma, CN2means, CN2sigmas, CNV, clusterID, matplotFile)


#####################
# Given all called CNVs for one sample, produce a PDF file in plotDir with plots for
# the relevant exons.
# CNVs must correspond to a single sample and be sorted by chrom-start.
def plotCNVsOneSample(CNVs, sampleID, exons, Ecodes, exonFPMs, samplesOfInterest,
                      isHaploid, CN0sigma, CN2means, CN2sigmas, clusterID, plotDir):
    plotFile = os.path.join(plotDir, sampleID + '_' + clusterID + '.pdf')
    matplotFile = matplotlib.backends.backend_pdf.PdfPages(plotFile)
    sampleIndex = CNVs[0][4]
    for CNV in CNVs:
        if (CNV[4] != sampleIndex):
            raise Exception("plotCNVsOneSample sanity: called with different samples")
        plotExonsOneCNV(CNV, sampleID, exons, Ecodes, exonFPMs, samplesOfInterest,
                        isHaploid, CN0sigma, CN2means, CN2sigmas, clusterID, matplotFile)
    matplotFile.close()
