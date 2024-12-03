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


###############################################################################################
##################################### test_exonViewer #########################################
###############################################################################################
# uses fragment count data, sample clustering information, and calculated parameters from
# various continuous distributions to generate coverage profiles.
# This tool enables the visualization of coverage profiles for a specific exon ID and
# a given sample given by a user TSV file.
# The resulting plots are saved in PDF format for further analysis and sharing.
# See usage for more details.
###############################################################################################
import getopt
import gzip
import logging
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import numpy
import os
import sys
import time

####### JACNEx modules
import countFrags.countsFile
import clusterSamps.clustFile
import callCNVs.exonProcessing
import CNVCalls.likelihoods

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS ################################
###############################################################################

####################################################
# parseArgs:
# Parse and sanity check the command+arguments, provided as a list of
# strings (eg sys.argv).
# Return a list with everything needed by this module's main()
# If anything is wrong, raise Exception("EXPLICIT ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    clustsFile = ""
    tocheckFile = ""
    pdfFile = ""

    usage = "NAME:\n" + scriptName + """\n

DESCRIPTION:
Given  three TSV files: one for fragment count data, another for sample cluster information,
and a user-defined file containing two columns [exonID, sampleID] tab-separated.
For each exon, it identifies the associated filtering criteria.
If the filtering value is 0 ("notCaptured") or 1 ("cannotFitRG"), a simple histogram will be plotted.
In cases where filtering is 2 ("RGClose2LowThreshold"), 3 ("fewSampsInRG"), or 4 ("call"),
specific distributions will be fitted for each copy number (CN0: half normal, CN1: Gaussian,
CN2: Gaussian, CN3: gamma) based on the FPM (fragments per million) distribution for the cluster
containing the target sample.
The resulting visualization includes a combined histogram and distribution plot with their respective parameters.
All graphical representations will be consolidated and stored as a single output PDF file.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --tocheck [str]: user-generated TSV file containing two columns [exonID(str), sampleID(str)] separated by tabs.
    --pdf[str]: a pdf file in which the graphical output will be produced.
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "tocheck=", "pdf="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
        elif (opt in ("--clusts")):
            clustsFile = value
        elif (opt in ("--tocheck")):
            tocheckFile = value
        elif (opt in ("--pdf")):
            pdfFile = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    if clustsFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(clustsFile)):
        raise Exception("clustsFile " + clustsFile + " doesn't exist.")

    if tocheckFile == "":
        raise Exception("you must provide a user file use --tocheck. Try " + scriptName + " --help.")
    elif (not os.path.isfile(tocheckFile)):
        raise Exception("tocheckFile " + tocheckFile + " doesn't exist.")

    if pdfFile == "":
        raise Exception("you must provide an pdfFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(pdfFile):
        raise Exception("pdfFile " + pdfFile + " already exists")
    elif (os.path.dirname(pdfFile) != '') and (not os.path.isdir(os.path.dirname(pdfFile))):
        raise Exception("the directory where pdfFile " + pdfFile + " should be created doesn't exist")

    # AOK, return everything that's needed
    return(countsFile, clustsFile, tocheckFile, pdfFile)


#########################
# parseUserTSV
# Parses a TSV file to map sample IDs to their respective exon indexes for autosomal and gonosomal exons.
#
# Args:
# - userTSV (str): Path to the user TSV file.
# - samples (list[str]): List of sample names.
# - autosomeExons (list of lists[str, int, int, str]): autosomal exon information [CHR, START, END, EXONID].
# - gonosomeExons (list of lists[str, int, int, str]): gonosomal exon information.
#
# Returns a tuple (samps2ExonInd_A, samps2ExonInd_G):
# - samps2ExonInd_A (dict): key==sampleID, value==List of exon indexes matching autosomal exons.
# - samps2ExonInd_G (dict): same as samps2ExonInd_A for gonosomes
def parseUserTSV(userTSV, samples, autosomeExons, gonosomeExons):

    if not os.path.exists(userTSV):
        raise FileNotFoundError(f"User TSV file {userTSV} does not exist.")

    sample_set = set(samples)
    autosome_dict = {exon[3]: idx for idx, exon in enumerate(autosomeExons)}
    gonosome_dict = {exon[3]: idx for idx, exon in enumerate(gonosomeExons)}

    samps2ExonInd_A, samps2ExonInd_G = {}, {}

    with gzip.open(userTSV, "rt") if userTSV.endswith(".gz") else open(userTSV) as file:
        for line in file:
            sID, exENST = line.rstrip().split("\t", maxsplit=1)
            if sID not in sample_set:
                logger.error(f"Sample name {sID} not found in provided samples list.")
                continue

            ei_A = autosome_dict.get(exENST)
            ei_G = gonosome_dict.get(exENST) if ei_A is None else None

            if ei_A is not None:
                samps2ExonInd_A.setdefault(sID, []).append(ei_A)
            elif ei_G is not None:
                samps2ExonInd_G.setdefault(sID, []).append(ei_G)
            else:
                logger.error(f"Exon identifier {exENST} not found in provided exon lists.")

    return (samps2ExonInd_A, samps2ExonInd_G)


###########################################################################
# createExonPlotsPDF
# Main function to create PDF files containing exon plots.
#
# Args:
# - pdfFile [str]: Path to the PDF file where plots will be saved.
# - samps2ExonInd_A (dict): key==sampleID, value==List of exon indexes matching autosomal exons.
# - samps2ExonInd_G (dict): same as samps2ExonInd_A for gonosomes
# - autosomeFPMs (numpy.ndarray[floats]): exon FPMs for autosomes. dim=[NBofExons, NBofSamps]
# - gonosomeFPMs (numpy.ndarray[floats])
# - intergenicFPMs (numpy.ndarray[floats])
# - autosomeExons (list of lists[str, int, int, str]): autosomal exon information.
# - gonosomeExons (list of lists[str, int, int, str])
# - other_params: Additional parameters required for plotting.
def createExonPlotsPDF(pdfFile, samps2ExonInd_A, samps2ExonInd_G, autosomeFPMs, gonosomeFPMs, intergenicFPMs,
                       autosomeExons, gonosomeExons, other_params):
    hnorm_loc, hnorm_scale, uncaptThreshold = computeCN0Parameters(intergenicFPMs)

    other_params['hnorm_loc'] = hnorm_loc
    other_params['hnorm_scale'] = hnorm_scale
    other_params['uncaptThreshold'] = uncaptThreshold

    if len(samps2ExonInd_A) != 0:
        createPlotsForChromosomeType(0, samps2ExonInd_A, autosomeFPMs, autosomeExons, "_autosomes.pdf", pdfFile, other_params)
    elif len(samps2ExonInd_G) != 0:
        createPlotsForChromosomeType(1, samps2ExonInd_G, gonosomeFPMs, gonosomeExons, "_gonosomes.pdf", pdfFile, other_params)
    else:
        raise


##########################################
# computeCN0Parameters
# Computes parameters for CN0 (Copy Number 0) based on intergenic FPMs.
#
# Args:
# - intergenicFPMs (numpy.ndarray[floats]): Array of FPMs for intergenic regions.
#
# Returns a tuple containing hnorm_loc (half Gaussian mean), hnorm_scale (half Gaussian standard deviation),
# and uncaptThreshold (threshold for uncaptured FPMs).
def computeCN0Parameters(intergenicFPMs):
    try:
        (hnorm_loc, hnorm_scale, uncaptThreshold) = callCNVs.exonProcessing.computeCN0Params(intergenicFPMs)
        return hnorm_loc, hnorm_scale, uncaptThreshold
    except Exception as e:
        logger.error("computeCN0Params failed: %s", str(e))
        raise


##########################################
# createPlotsForChromosomeType
# Creates exon plots for a specific chromosome type (autosomes or gonosomes).
#
# Args:
# - chromType [int]: Type of chromosome (0='autosomes' or 1='gonosomes').
# - samps2ExonInd (dict): key==sampleID, value==List of exon indexes
# - FPMs (numpy.ndarray[floats])
# - exons (list of lists[str, int, int, str])
# - pdfFileSuffix [str]: Suffix for the PDF file name.
# - pdfFile [str]: Path to the base PDF file.
# - other_params: Additional parameters for plotting.
def createPlotsForChromosomeType(chromType, samps2ExonInd, FPMs, exons, pdfFileSuffix, pdfFile, other_params):
    try:
        pdfFile = generatePDFRootName(pdfFile, pdfFileSuffix)
        generateClusterExonPlots(chromType, samps2ExonInd, FPMs, exons, other_params, pdfFile)
    except Exception as e:
        logger.error(f"generateClusterExonPlots failed for {chromType}: {str(e)}")
        raise


#####################################
# Generate root name for output PDF files.
# returns a new PDF file name with the given suffix.
def generatePDFRootName(pdfFile, suffix):
    rootFileName = os.path.splitext(pdfFile)[0]
    return rootFileName + suffix


######################################
# generateClusterExonPlots
# Generates exon plots for a cluster of samples.
#
# Args:
# - chromType [int]: Type of chromosome.
# - samps2Check (dict): key==sampleID, value==List of exon indexes
# - FPMs (numpy.ndarray[floats])
# - exons (list of lists[str, int, int, str])
# - other_params: Additional parameters required for plotting.
# - pdfFile: Path to the PDF file for saving plots.
def generateClusterExonPlots(chromType, samps2Check, FPMs, exons, other_params, pdfFile):
    matplotlib.pyplot.close('all')
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdfFile)
    samp2Index = {sample: i for i, sample in enumerate(other_params['samples'])}

    try:
        for sampID, exonIndexes in samps2Check.items():
            plotExonsForSample(sampID, exonIndexes, chromType, samp2Index, FPMs, exons, other_params, matplotOpenFile)
    except Exception as e:
        logger.error(f"Error processing exon plots: {e}")
    finally:
        matplotOpenFile.close()


##################################
# plotExonsForSample
# Handles plotting of exons for a specific sample.
#
# Args:
# - sampID [str]
# - exonIndexes (list[int])
# - chromType [int]
# - samp2Index(dict): key==sampleID, value==List of exon indexes
# - FPMs (numpy.ndarray[floats])
# - exons (list of lists[str, int, int, str])
# - other_params: Additional parameters required for plotting.
# - matplotOpenFile: Matplotlib PDF file object for saving plots.
def plotExonsForSample(sampID, exonIndexes, chromType, samp2Index, FPMs, exons, other_params, matplotOpenFile):
    samp2clusts = other_params['samp2clusts']
    clust2samps = other_params['clust2samps']
    fitWith = other_params['fitWith']
    unCaptFPMLimit = other_params['uncaptThreshold']
    hnorm_loc = other_params["hnorm_loc"]
    hnorm_scale = other_params["hnorm_scale"]

    clusterID = samp2clusts[sampID][chromType]
    exonIndexes = numpy.array(exonIndexes)

    sampsInClust = callCNVs.exonProcessing.getSampIndexes(clusterID, clust2samps, samp2Index, fitWith)
    sampsInClust = numpy.array(sampsInClust)

    targetFPMs = FPMs[exonIndexes[:, None], sampsInClust]
    CN2params, filterStatesVec = computeExonCNVParameters(targetFPMs, unCaptFPMLimit)

    for ei, realExInd in enumerate(exonIndexes):
        if filterStatesVec[ei] < 2:
            logger.warning(f"Skipping plot for exon index {realExInd}, sample {sampID}, cluster {clusterID}")
            continue

        try:
            getReadyPlotExon(sampID, realExInd, clusterID, FPMs[realExInd, samp2Index[sampID]], targetFPMs[ei],
                             CN2params[ei], filterStatesVec[ei], exons[realExInd], hnorm_loc, hnorm_scale, matplotOpenFile)
        except Exception as e:
            logger.error(f"Plotting failed for exon index {realExInd}, sample {sampID}, cluster {clusterID}: {e}")
            continue


############################################
# computeExonCNVParameters
# Computes parameters for exon Copy Number Variation (CNV) analysis.
#
# Args:
# - targetFPMs (numpy.ndarray[floats]): Target FPM values for exons across cluster samples.
# - uncaptThreshold [float]: Threshold value for filtering out uncaptured FPMs.
#
# Returns a tuple containing CN2ParamsArray (array of CN2 parameters for each exon,
# including mean and standard deviation) and filterStatesVec (array of filter state indices for each exon).
def computeExonCNVParameters(targetFPMs, uncaptThreshold):
    filterStates = ["notCaptured", "cannotFitRG", "RGClose2LowThreshold", "fewSampsInRG", "call"]
    paramsID = ["loc", "scale"]
    CN2ParamsArray = numpy.full((targetFPMs.shape[0], len(paramsID)), -1, dtype=numpy.float64)
    filterStatesVec = numpy.zeros(targetFPMs.shape[0], dtype=int)

    for exonIndex, exFPMs in enumerate(targetFPMs):
        try:
            filterState, exonCN2Params = callCNVs.exonProcessing.evaluateAndComputeExonMetrics(exFPMs, uncaptThreshold, paramsID)
            filterStatesVec[exonIndex] = filterStates.index(filterState)
            CN2ParamsArray[exonIndex] = exonCN2Params
        except Exception as e:
            logger.error(f"Error processing exon metrics: {e}")
            raise

    return (CN2ParamsArray, filterStatesVec)


######################################
# getReadyPlotExon
# Parses and plots exon-related data for a specific sample and exon.
#
# Args:
# - sampID [str]
# - ei [int]
# - clusterID [int]
# - sampExFPM [float]: sample exon FPM
# - exClustFPMs (numpy.ndarray[floats]): exon FPM values for all samples in clusterID
# - exCN2params (numpy.ndarray[floats]): exon gaussian parameters [loc, scale]
# - exfilterState [int]: filter index
# - exInfos (list[str, int, int, str]): exon [CHR, START, END, EXONID]
# - hnorm_loc (float): distribution location parameter.
# - hnorm_scale (float): distribution scale parameter.
# - matplotOpenFile: Opened PDF file for plotting.
def getReadyPlotExon(sampID, ei, clusterID, sampExFPM, exClustFPMs, exCN2params, exfilterState, exInfos,
                     hnorm_loc, hnorm_scale, matplotOpenFile):
    ##### init graphic parameters
    # definition of a range of values that is sufficiently narrow to have
    # a clean representation of the adjusted distributions.
    xi = numpy.linspace(0, numpy.max(exClustFPMs), len(exClustFPMs) * 3)

    exonInfo = '_'.join(map(str, exInfos))
    plotTitle = f"Cluster:{clusterID}, NbSamps:{len(exClustFPMs)}, exonInd:{ei}\nexonInfos:{exonInfo}, filteringState:{exfilterState}"

    yLists = []
    plotLegs = []
    verticalLines = [sampExFPM]
    vertLinesLegs = f"{sampID} FPM={sampExFPM:.3f}"

    # get gauss_loc and gauss_scale from exParams
    gaussLoc = exCN2params[0]
    gaussScale = exCN2params[1]

    # Update plot title with likelihood information
    plotTitle += f"\n{sampID} likelihoods:\n"

    distribution_functions = CNVCalls.likelihoods.getDistributionObjects(hnorm_loc, hnorm_scale, gaussLoc, gaussScale)

    for ci in range(len(distribution_functions)):
        pdf_function, loc, scale, shape = distribution_functions[ci]
        # numpy.ndarray 1D float: set of pdfs for all samples
        # scipy execution speed up
        if shape is not None:
            PDFRanges = pdf_function(xi, loc=loc, scale=scale, a=shape)
            sampLikelihood = pdf_function(sampExFPM, loc=loc, scale=scale, a=shape)
            plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}, shape={shape:.2f}]")
        else:
            PDFRanges = pdf_function(xi, loc=loc, scale=scale)
            sampLikelihood = pdf_function(sampExFPM, loc=loc, scale=scale)
            plotLegs.append(f"CN{ci} [loc={loc:.2f}, scale={scale:.2f}]")

        yLists.append(PDFRanges)
        plotTitle += f"CN{ci}:{sampLikelihood:.3e} "

    ylim = 2 * numpy.max(yLists[2])

    plotExonProfile(exClustFPMs, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, matplotOpenFile)


#########################
# plotExonProfile
# plots a density histogram for a raw data list, along with density or distribution
# curves for a specified number of data lists. It can also plot vertical lines to mark
# points of interest on the histogram. The graph is saved as a PDF file.
#
# Args:
# - rawData (numpy.ndarray[float]): exon FPM counts
# - xi (list[float]): x-axis values for the density or distribution curves, ranges
# - yLists (list of lists[float]): y-axis values, probability density function values
# - plotLegs (list[str]): labels for the density or distribution curves
# - verticalLines (list[float]): vertical lines to be plotted, FPM tresholds
# - vertLinesLegs (list[str]): labels for the vertical lines to be plotted
# - plotTitle [str]: title of the plot
# - pdf (matplotlib.backends object): a file object for save the plot
def plotExonProfile(rawData, xi, yLists, plotLegs, verticalLines, vertLinesLegs, plotTitle, ylim, pdf):
    # Disable interactive mode to prevent display of the plot during execution
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(8, 8))
    # Plot a density histogram of the raw data with a number of bins equal to half the number of data points
    matplotlib.pyplot.hist(rawData, bins=int(len(rawData) / 2), density=True)

    # Plot the density/distribution curves for each set of x- and y-values
    if len(yLists) > 1:
        for i in range(len(yLists)):
            matplotlib.pyplot.plot(xi, yLists[i], label=plotLegs[i])

    # Plot vertical lines to mark points of interest on the histogram
    if verticalLines:
        matplotlib.pyplot.axvline(verticalLines, color="red", linestyle='dashdot', linewidth=1, label=vertLinesLegs)

    # Set the x- and y-axis labels, y-axis limits, title, and legend
    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("probability density (=likelihoods)")
    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.title(plotTitle)
    matplotlib.pyplot.legend(loc='upper right', fontsize='small')

    pdf.savefig(fig)
    matplotlib.pyplot.close()


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):

    # parse, check and preprocess arguments
    (countsFile, clustsFile, tocheckFile, pdfFile) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse and FPM-normalize the counts, distinguishing between exons and intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("Failed to parse and normalize counts for %s: %s", countsFile, str(e))
        raise Exception("Failed to parse and normalize counts")

    thisTime = time.time()
    logger.debug("Done parsing and normalizing counts file, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ###################
    # parse clusters informations
    try:
        (clust2samps, samp2clusts, fitWith, clust2gender, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clustsFile)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, str(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # parse user file
    try:
        (samps2ExonInd_A, samps2ExonInd_G) = parseUserTSV(tocheckFile, samples, autosomeExons, gonosomeExons)
    except Exception as e:
        raise Exception("parseUserTSV failed for %s : %s", tocheckFile, str(e))

    thisTime = time.time()
    logger.debug("Done parsing UserTSV, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ####################
    # plot exon profil
    try:
        other_params = {'samples': samples, 'clust2samps': clust2samps, 'samp2clusts': samp2clusts, 'fitWith': fitWith}
        createExonPlotsPDF(pdfFile, samps2ExonInd_A, samps2ExonInd_G, autosomeFPMs, gonosomeFPMs, intergenicFPMs,
                           autosomeExons, gonosomeExons, other_params)
    except Exception as e:
        raise Exception("generateExonPlots failed for %s : %s", tocheckFile, str(e))

    thisTime = time.time()
    logger.debug("Done generateExonPlots, in %.2f s", thisTime - startTime)
    startTime = thisTime

    thisTime = time.time()
    logger.debug("Done exon profil plots, in %.2f s", thisTime - startTime)
    startTime = thisTime


####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    scriptName = os.path.basename(sys.argv[0])
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(scriptName)

    try:
        main(sys.argv)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + scriptName + " : " + str(e) + "\n")
        sys.exit(1)
