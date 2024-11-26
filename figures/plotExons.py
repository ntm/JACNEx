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


import numpy
import math
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_pdf
import logging
import os
import datetime

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
def plotExons(exons, exonsToPlot, Ecodes, exonFPMs, samplesOfInterest, isHaploid, CN0sigma, CN2means, CN2sigmas, clusterID, plotDir):
    # return immediately if exonsToPlot is empty
    if not exonsToPlot:
        return

    # construct the filename
    current_time = datetime.datetime.now().strftime("%y%m%d_%H-%M-%S")
    numExons = len(exonsToPlot)
    plotFile = os.path.join(plotDir, f'{clusterID}_plotExons_{numExons}_{current_time}.pdf')
    matplotlib.use('pdf')
    matplotFile = matplotlib.backends.backend_pdf.PdfPages(plotFile)

    # number of bins for the histograms: try to use numSamples/2, but at least 40 and at most 100
    bins = int(min(max(40, exonFPMs.shape[1] / 2), 100))

    for thisExon in exonsToPlot.keys():
        # skip filtered exons : NOT-CAPTURED and FIT-CN2-FAILED
        if (Ecodes[thisExon] == -1) or (Ecodes[thisExon] == -2):
            logging.debug(f"Skipping exon {thisExon} ({exons[thisExon][0]}:{exons[thisExon][1]}-{exons[thisExon][2]} {exons[thisExon][3]}), Status: {Ecodes[thisExon]}")
            continue

        fpms = exonFPMs[thisExon, :]

        # on x axis: plot up to 20% beyond max FPM
        fpmMax = max(fpms) * 1.2
        # 1000 evenly-spaced x coordinates, for smooth plots
        xcoords = numpy.linspace(0, fpmMax, 1000)

        pdfs = callCNVs.likelihoods.calcLikelihoods(
            xcoords.reshape(1, -1), CN0sigma, numpy.array([Ecodes[thisExon]]),
            numpy.array([CN2means[thisExon]]), numpy.array([CN2sigmas[thisExon]]),
            isHaploid, True)

        labels = getLabels(isHaploid, CN0sigma, CN2means[thisExon], CN2sigmas[thisExon])

        fpmSOI = fpms[samplesOfInterest]
        fpmNonSOI = fpms[~samplesOfInterest]

        plotHistogramAndPdfs(fpmSOI, fpmNonSOI, bins, xcoords, pdfs, exons, thisExon, clusterID,
                             Ecodes, exonsToPlot, labels, isHaploid, matplotFile)
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
# Plot histogram and PDFs for an exon
def plotHistogramAndPdfs(fpmSOI, fpmNonSOI, bins, xcoords, pdfs, exons, thisExon, clusterId, ECodes,
                         exonsToPlot, labels, isHaploid, matplotFile):

    colors = ['red', 'orange', 'green', 'purple']
    ECodeSTR = {1: 'CALLED-WITHOUT-CN1', 0: 'CALLED', -3: 'CN2-LOW-SUPPORT', -4: 'CN0-TOO-CLOSE'}

    if isHaploid:
        limY = 2 * max(pdfs[:, :, 2])
    else:
        limY = 1.2 * max(pdfs[:, :, 1])

    fig = matplotlib.figure.Figure(figsize=(8, 6))
    ax = fig.subplots()

    yhist = ax.hist(fpmSOI,
                    bins=bins,
                    range=(0, xcoords[-1]),
                    label='FPMs (this cluster)',
                    density=True,
                    color='lightblue',
                    edgecolor='black')[0]
    limY = max(limY, 1.2 * max(yhist))

    # histogram of FITWITH samples: on top, but with transparency
    if len(fpmNonSOI) != 0:
        ax.hist(fpmNonSOI,
                bins=bins,
                range=(0, xcoords[-1]),
                label='FPMs (FITWITH cluster)',
                density=True,
                color='grey',
                edgecolor='black',
                alpha=0.5)

    # plot CN models
    for cnState in range(4):
        if (labels[cnState] != ''):
            if (ECodes[thisExon] == 1) and (cnState == 1):
                linestyleFIT = 'dashed'
            else:
                linestyleFIT = 'solid'
            ax.plot(xcoords,
                    pdfs[:, :, cnState].flatten(),
                    linewidth=3,
                    linestyle=linestyleFIT,
                    color=colors[cnState],
                    label=labels[cnState])

    # vertical line(s) for target sample(s)
    sampleColors = matplotlib.colormaps['tab20'](numpy.linspace(0, 1, len(exonsToPlot[thisExon])))

    for sampleInfo, sampleColor in zip(exonsToPlot[thisExon], sampleColors):
        sampleFpm = fpmSOI[sampleInfo[0]]
        ax.axvline(sampleFpm,
                   color=sampleColor,
                   linewidth=3,
                   linestyle='dashed',
                   label=f'{sampleInfo[1]} ({sampleFpm:.2f} FPM)')

    title_text = (f"{clusterId}\n"
                  f"{exons[thisExon][0]}:{exons[thisExon][1]}-{exons[thisExon][2]} {exons[thisExon][3]}\n"
                  f"{ECodeSTR.get(ECodes[thisExon], 'UNKNOWN')}")
    ax.set_title(title_text)
    ax.set_xlabel("FPM")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, xcoords[-1])
    ax.set_ylim(0, limY)
    matplotFile.savefig(fig)
