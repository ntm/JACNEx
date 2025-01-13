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


###########################################################################
# Given a countsFile made by s1_countFrags.py, produce density plots of the
# FPMs for each sample, and identify any atypical "QC-failing" samples.
# See usage for details.
###########################################################################
import getopt
import KDEpy
import logging
import matplotlib.backends.backend_pdf
import numpy
import os
import sys

####### JACNEx modules
import countFrags.countsFile
import devel.plots

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


###############################################################################
############################ PRIVATE FUNCTIONS ################################
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
    # optional args with default values
    plotDir = "./plotDir/"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
With a typical exome, the density plot of FPM counts over all exons appears as a mixture of:
A) a sharp peak at 0 FPM (eg exons that are not captured by the exome kit)
B) a vaguely bell-shaped curve with a heavy tail (captured exons).
This script identifies and logs any atypical "QC-failing" samples to stderr, and
produces FPM density plots for PASSing and FAILing samples.

ARGUMENTS:
   --counts [str]: TSV file of fragment counts, possibly gzipped, produced by s1_countFrags.py
   --plotDir [str]: subdir (created if needed) where plot files will be produced, default:  """ + plotDir + """
   -h , --help : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "plotDir="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
        elif (opt in ("--plotDir")):
            plotDir = value
        else:
            raise Exception("unhandled option " + opt)

    # Check args
    if countsFile == "":
        raise Exception("you must provide a countsFile with --counts. Try " + scriptName + " --help")
    elif not os.path.isfile(countsFile):
        raise Exception("countsFile " + countsFile + " doesn't exist")

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, plotDir)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# Produce plots as pdf files in plotDir, and report QC-failing samples on stderr.
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    (countsFile, plotDir) = parseArgs(argv)

    logger.info("starting to work")

    ###################
    # parse and FPM-normalize the counts, distinguishing between exons and intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, str(e))
        raise Exception("parseAndNormalizeCounts failed")

    logger.info("done parseAndNormalizeCounts")

    ###################
    # plot exon FPM densities for all samples; use this to identify QC-failing samples

    # should density plots compare several different KDE bandwidth algorithms and values?
    # hard-coded here rather than set via parseArgs because this should only be set
    # to True for dev & testing
    testSmoothingBWs = False

    plotFilePass = plotDir + "/QC_FPMs_PASS.pdf"
    plotFileFail = plotDir + "/QC_FPMs_FAIL.pdf"
    try:
        SampsQC(autosomeFPMs, samples, plotFilePass, plotFileFail, testBW=testSmoothingBWs)
    except Exception as e:
        logger.error("SampsQC failed for %s : %s", countsFile, str(e))
        raise Exception("SampsQC failed")

    logger.info("ALL DONE")


###################################
# SampsQC :
# With a typical exome, the density plot of FPM counts over all exons appears as
# a mixture of:
# A) a sharp peak at 0 FPM (eg exons that are not captured by the exome kit)
# B) a vaguely bell-shaped curve with a heavy tail (captured exons).
# This function identifies any atypical "QC-failing" samples, and produces FPM
# density plots for PASSing and FAILing samples.
#
# Args:
# - counts (numpy.ndarray[float]): normalised fragment counts
# - samples (list[str]): sample names
# - plotFilePass: pdf filename (with path) where plots for QC-passing samples are made
# - plotFileFail: same as plotFilePass but for QC-failing samples
# - minLow2high (float): a sample's FPM density must increase by a fraction at least
#     minLowToHigh between it's first local min and it's subsequent max to pass QC
# - testBW: if True each plot includes several different KDE bandwidth algorithms/values,
#     for testing and comparing; otherwise use what seems best in our tests
#
# Return sampsQCfailed:
#  - sampsQCfailed: list of indexes (in samples) of samples that fail QC
def SampsQC(counts, samples, plotFilePass, plotFileFail, minLow2high=0.2, testBW=False):
    if os.path.isfile(plotFilePass) or os.path.isfile(plotFileFail):
        logger.info('SampsQC : pre-exsiting plotFile(s) %s and/or %s will be squashed',
                    plotFilePass, plotFileFail)

    sampsQCfailed = []

    # create matplotlib PDF objects
    pdfPass = matplotlib.backends.backend_pdf.PdfPages(plotFilePass)
    pdfFail = matplotlib.backends.backend_pdf.PdfPages(plotFileFail)

    for sampleIndex in range(len(samples)):
        sampleCounts = counts[:, sampleIndex]

        # for smoothing and plotting we don't care about large counts (and they mess things up),
        # we will only consider the bottom fracDataForSmoothing fraction of counts, default
        # value should be fine
        fracDataForSmoothing = 0.96
        # corresponding max counts value
        maxData = numpy.quantile(sampleCounts, fracDataForSmoothing)

        # produce smoothed representation(s) (one if testBW==False, several otherwise)
        dataRanges = []
        densities = []
        legends = []
        # vertical dashed lines: coordinates (default to 0 ie no lines) and legends
        (xmin, xmax) = (0, 0)
        (line1legend, line2legend) = ("", "")
        # pdf to use for this sample
        pdf = pdfPass

        if testBW:
            allBWs = ('ISJ', 0.15, 0.20, 'scott', 'silverman')
        else:
            # our current favorite, gasp at the python-required trailing comma...
            allBWs = ('ISJ',)

        for bw in allBWs:
            try:
                (dr, dens, bwValue) = smoothData(sampleCounts, maxData=maxData, bandwidth=bw)
            except Exception as e:
                logger.error('smoothing failed for %s : %s', str(bw), str(e))
                raise
            dataRanges.append(dr)
            densities.append(dens)
            legend = str(bw)
            if isinstance(bw, str):
                legend += ' => bw={:.2f}'.format(bwValue)
            legends.append(legend)

        # find indexes of first local min density and first subsequent local max density,
        # looking only at our first smoothed representation
        try:
            (minIndex, maxIndex) = findFirstLocalMinMax(densities[0])
        except Exception as e:
            logger.info("sample %s is bad: %s", samples[sampleIndex], str(e))
            sampsQCfailed.append(sampleIndex)
            pdf = pdfFail
        else:
            (xmin, ymin) = (dataRanges[0][minIndex], densities[0][minIndex])
            (xmax, ymax) = (dataRanges[0][maxIndex], densities[0][maxIndex])
            line1legend = "min FPM = " + '{:0.2f}'.format(xmin)
            line2legend = "max FPM = " + '{:0.2f}'.format(xmax)
            # require at least minLow2high fractional increase from ymin to ymax
            if ((ymax - ymin) / ymin) < minLow2high:
                logger.info("sample %s is bad: min (%.2f,%.2f) and max (%.2f,%.2f) densities too close",
                            samples[sampleIndex], xmin, ymin, xmax, ymax)
                sampsQCfailed.append(sampleIndex)
                pdf = pdfFail

        if pdf == pdfFail:
            # need ymax for plotting even if sample is bad, the max Y after the first 15%
            # of dataRange is a good rule of thumb (not too cropped, not too zoomed out)
            ymax = max(densities[0][int(len(dataRanges[0]) * 0.15):])

        # plot all the densities for sampleIndex in a single plot
        title = samples[sampleIndex] + " density of exon FPMs"
        # max range on Y axis for visualization, 3*ymax should be fine
        ylim = 3 * ymax
        devel.plots.plotDensities(title, dataRanges, densities, legends, xmin, xmax, line1legend, line2legend, ylim, pdf)

    # close PDFs
    pdfPass.close()
    pdfFail.close()

    if len(sampsQCfailed) == len(samples):
        logger.warning("all samples FAIL the FPM density QC test, the test may be inappropriate for your data?")
        os.unlink(plotFilePass)
    elif len(sampsQCfailed) > 0:
        logger.warning("%s/%s samples fail FPM density QC (check the plots), these samples may get unreliable CNV calls",
                       len(sampsQCfailed), len(samples))
    else:
        logger.info("all %s samples pass FPM density QC, great!", len(samples))
        os.unlink(plotFileFail)

    return(sampsQCfailed)


###################################
# Given a vector of floats, apply kernel density estimation to obtain a smoothed
# representation of the data.
#
# Args:
# - data: a 1D numpy.ndarray of floats (>= 0)
# - maxData (float): data values beyond maxData are not plotted (but they ARE
#     used to calculate the bandwidth when using a heuristic)
# - numValues (int): approximate size of the returned x and y, default should be fine
# - bandwidth: one of ('scott', 'silverman', 'ISJ') to use as bandwidth determination
#     heuristic, or a float (to use a fixed bandwidth)
#
# Returns a tuple (x, y, bwValue):
# - x and y are numpy.ndarrays of the same size representing the smoothed density of data
# - bwValue is the bandwidth value (float) used for the KDE
def smoothData(data, maxData=10, numValues=1000, bandwidth='scott'):
    # if using a bandwidth heuristic ('scott', 'silverman' or 'ISJ'):
    # calculate bwValue using full dataset
    if isinstance(bandwidth, str):
        kde = KDEpy.FFTKDE(bw=bandwidth)
        kde.fit(data).evaluate()
        bwValue = kde.bw
    else:
        bwValue = bandwidth

    # allow 3 bwValues (==stddev since we use gaussian kernels) beyond maxData,
    # to avoid artifactual "dip" of the curve close to maxData
    dataReduced = data[data <= maxData + 3 * bwValue]

    # we have a hard lower limit at 0, this causes some of the weight of close-to-zero
    # values to be ignored (because that weight goes to negative values)
    # this issue can be alleviated by mirroring the data, see:
    # https://kdepy.readthedocs.io/en/latest/examples.html#boundary-correction-using-mirroring
    dataReducedMirrored = numpy.concatenate((dataReduced, -dataReduced))
    # Compute KDE using bwValue, and twice as many grid points
    (dataRange, density) = KDEpy.FFTKDE(bw=bwValue).fit(dataReducedMirrored).evaluate(numValues * 2)
    # double the y-values to get integral of ~1 on the positive dataRanges
    density = density * 2
    # delete values outside of [0, maxData]
    density = density[numpy.logical_and(0 <= dataRange, dataRange <= maxData)]
    dataRange = dataRange[numpy.logical_and(0 <= dataRange, dataRange <= maxData)]
    return(dataRange, density, bwValue)


###################################
# find:
# - the first local min of data, defined as the first data value such that
#   the next windowSize-1 values of data are > data[minIndex] (ignoring
#   stretches of equal values)
# - the first local max of data after minIndex, with analogous definition
#
# Args:
# - data: a 1D numpy.ndarray of floats
# - windowSize: int, default should be fine if running on y returned by smoothData()
#     with its default numValues
#
# Returns a tuple: (minIndex, maxIndex)
#
# Raise exception if no local min / max is found (eg data is always decreasing, or
# always increasing after minIndex)
def findFirstLocalMinMax(data, windowSize=20):
    # sanity checks
    if windowSize <= 0:
        logger.error("in findFirstLocalMinMax, windowSize must be a positive int, not %s", str(windowSize))
        raise Exception('findLocalMinMax bad args')
    if windowSize > len(data):
        logger.error("findFirstLocalMinMax called with windowSize > len(data), useless")
        raise Exception('findLocalMinMax data too small')

    # find first local min
    minIndex = 0
    minValue = data[minIndex]
    # number of consecutive values >= minValue seen
    thisWindowSize = 1

    for i in range(1, len(data)):
        if data[i] < minValue:
            minIndex = i
            minValue = data[i]
            thisWindowSize = 1
        elif data[i] > minValue:
            thisWindowSize += 1
            if thisWindowSize >= windowSize:
                break
        # else data[i] == minValue, look further

    if thisWindowSize < windowSize:
        raise Exception('findLocalMinMax cannot find a local min')

    # find first local max following minIndex
    maxIndex = minIndex
    maxValue = minValue
    thisWindowSize = 1
    for i in range(minIndex + 1, len(data)):
        if data[i] > maxValue:
            maxIndex = i
            maxValue = data[i]
            thisWindowSize = 1
        elif data[i] < maxValue:
            thisWindowSize += 1
            if thisWindowSize >= windowSize:
                break
        # else continue

    if thisWindowSize < windowSize:
        raise Exception('findLocalMinMax cannot find a max after the min')

    return(minIndex, maxIndex)


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
