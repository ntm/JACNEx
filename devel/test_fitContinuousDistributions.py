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
# Given a TSV of exon fragment counts produced by 1_countFrags.py
# It applies various continuous distributions from scipy's "continuous_distns" to
# normalized fragments counts of intergenic regions.
# See usage for more details.
###############################################################################################
import getopt
import logging
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import numpy
import os
import scipy.stats
import sys
from scipy.stats._continuous_distns import _distn_names
import time
import warnings

####### JACNEx modules
import countFrags.countsFile

# prevent PIL and matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('PIL').setLevel(logging.WARNING)
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
    plotDir = "./plotDir/"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of intergenic fragment counts, fit this empirical data to various continuous
probability distributions available in the "scipy.stats" library of Python.
It adjusts continuous distributions to the input data and ranks the best-fitting distributions
based on their coefficient of determination (sum squared error).
The script can also generate plots to visualize both the actual data and the best-fitting
distributions.
Notably, the results of each fit are saved in the log file.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
                    hold the fragment counts. File obtained from 1_countFrags.py.
    --plotDir [str]: sub-directory in which the graphical PDFs will be produced, default:  """ + plotDir + """
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "plotDir="])
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
        elif (opt in ("--plotDir")):
            plotDir = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, plotDir)


###############################################################################
############################ PRIVATE FUNCTIONS #################################
###############################################################################
############################
# bestFitContinuousDistribs
# Given FPMs raw data, computes 106 continuous laws available in the
# scipy _distn_names library (v.3.9).
# Inspired by a discussion forum :
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
# The continuous distributions with more than 3 parameters were also excluded
# for user-friendliness simplification
#
# Args:
# - data (list[float]): raw data in FPM
# - bins [int]: Number of bins for histogram.
#
# return:
# - best_distributions (list of lists): list of lists containing
#                                       [DISTRIBNAME,[PARAMS],[PDF], SSE],
#                                       ordered by sse
def bestFitContinuousDistribs(data, bins):
    # Get histogram of original data
    y, x = numpy.histogram(data, bins=bins, density=True)
    x = (x + numpy.roll(x, -1))[:-1] / 2.0

    # Initialize output lists
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate(_distn_names):
        # Use getattr() to obtain the corresponding statistical distribution
        # object from the scipy.stats module
        distribution = getattr(scipy.stats, distribution)

        # Try to fit the current distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Compute distribution parameters from raw data, usually containing
                # loc and scale, but could have additional parameters
                params = distribution.fit(data)

                # Skip distributions with more than 3 parameters
                if len(params) > 3:
                    logger.info("{:>3} / {:<3}: {} NOT FIT: num args > 3".format(ii + 1, len(_distn_names), distribution.name))
                    continue

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)

                #######
                # Calculate SSE (Sum of Squared Errors)
                # its accuracy is limited as it heavily depends on user-defined bin choices
                # in a histogram.
                # It cannot substitute the user's visual assessment since variations in bins
                # can lead to significant changes in SSE without necessarily reflecting the
                # true quality of the fit.
                sse = numpy.sum((y - pdf)**2)

                if numpy.isnan(sse):
                    logger.info("{:>3} / {:<3}: {} NOT FIT: nan r² score".format(ii + 1, len(_distn_names), distribution.name))
                else:
                    best_distributions.append((distribution, params, pdf, sse))
                    logger.info("{:>3} / {:<3}: {} params:{}; sse:{}".format(ii + 1, len(_distn_names), distribution.name, params, sse))

        except Exception as e:
            logger.info("{:>3} / {:<3}: {} NOT FIT: {}".format(ii + 1, len(_distn_names), distribution.name, e))
            continue

    return sorted(best_distributions, key=lambda x: x[3], reverse=True)


########################
# plotDistrib
# Plots real data and best-fit probability distributions on the same graph.
#
# Args:
# - intergenicVec (list[float]): data points
# - userBins (int): Number of bins for histogram.
# - distributions (list): best-fit distributions to be plotted.
# - plotFile (str): Path to the output PDF file where the plot will be saved.
# - NBBestFit (int): Number of top best-fit distributions to include in the plot.
# - ylim (int)
def plotDistrib(intergenicVec, userBins, distributions, plotFile, NBBestFit, ylim):
    # Select the top NBBestFit distributions
    top = distributions[:NBBestFit]

    # Create a PDF file for the plot
    pdfFile = matplotlib.backends.backend_pdf.PdfPages(plotFile)
    fig = matplotlib.pyplot.figure(figsize=(16, 12))

    # Plot the real data as a histogram
    n, bins, patches = matplotlib.pyplot.hist(intergenicVec,
                                              bins=userBins,
                                              density=True,
                                              alpha=0.2,
                                              color='blue',
                                              label="Real Data")
    # Plot the best-fit distribution
    for i in range(len(top)):
        label = str(top[i][0].name)
        matplotlib.pyplot.plot(bins[:-1],
                               top[i][2],
                               label=f'{label} r²={numpy.round(top[i][4], 2)}',
                               linewidth=3.0,
                               linestyle='--')

    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.ylabel("Densities")
    matplotlib.pyplot.xlabel("FPMs")
    pdfFile.savefig(fig)
    pdfFile.close()


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
    (countsFile, plotDir) = parseArgs(argv)

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

    #################
    # preparing data for fit
    ######
    # Dev user params
    # limitation of samples number to be processed because the process is very long
    # aproximately 11min for 168,000 intergenic regions (= one sample from ensembl)
    NBSampsToProcess = 50
    userBins = 100
    NBBestFit = 10
    plotFile1 = os.path.join(plotDir, str(NBBestFit) + "_bestContinuousDistrib_" + str(userBins) + "bins_" + str(NBSampsToProcess) + "samps.pdf")
    plotFile2 = os.path.join(plotDir, str(NBBestFit) + "_bestContinuousDistrib_" + str(userBins) + "bins_" + str(NBSampsToProcess) + "_" + str(len(samples)) + "samps.pdf")
    plotFile3 = os.path.join(plotDir, str(NBBestFit) + "_bestContinuousDistrib_Only2params_" + str(userBins) + "bins_" + str(NBSampsToProcess) + "samps.pdf")
    plotFile4 = os.path.join(plotDir, str(NBBestFit) + "_bestContinuousDistrib_Only2params_" + str(userBins) + "bins_" + str(NBSampsToProcess) + "_" + str(len(samples)) + "samps.pdf")

    # Generate random indexes for the columns
    randomSampInd = numpy.random.choice(intergenicFPMs.shape[1], NBSampsToProcess, replace=False)
    sampleNames = ' '.join([samples[index] for index in randomSampInd])
    logger.info(f"samples analysed: {sampleNames}")

    # Select the corresponding columns intergenicFPMs
    randomCountsArray = intergenicFPMs[:, randomSampInd]

    # reshaping (e.g., -1 to keep the number of rows)
    vec = randomCountsArray.reshape(intergenicFPMs.shape[0], -1)

    # for plotting we don't care about large counts (and they mess things up),
    # we will only consider the bottom fracDataForSmoothing fraction of counts, default
    # value should be fine
    fracDataForSmoothing = 0.99
    # corresponding max counts value
    maxData = numpy.quantile(vec, fracDataForSmoothing)
    subFPMsVec = vec[vec <= maxData]

    interVec = intergenicFPMs.reshape(-1)
    maxData = numpy.quantile(interVec, fracDataForSmoothing)
    intergenicFPMsVec = interVec[interVec <= maxData]

    #####################
    # fitting continuous distributions
    try:
        distributions = bestFitContinuousDistribs(subFPMsVec, userBins)
    except Exception as e:
        raise Exception("bestFitContinuousDistribs failed: %s", str(e))

    thisTime = time.time()
    logger.debug("Done fitting continuous distributions on intergenic counts, in %.2f s", thisTime - startTime)
    startTime = thisTime

    #####################
    # Plot
    # Define a list of data and corresponding plot files
    data_files = [(subFPMsVec, plotFile1),
                  (intergenicFPMsVec, plotFile2),
                  (subFPMsVec, plotFile3),
                  (intergenicFPMsVec, plotFile4)]

    # Define the list of distributions to use
    topDistrib_2params = []
    for di in distributions:
        if len(di[1]) <= 2:
            topDistrib_2params.append(di)

    distributions_to_use = [distributions, distributions, topDistrib_2params, topDistrib_2params]
    ylim = [300, 50, 300, 50]

    # Iterate through the data files and distributions
    for i, (data, plot_file) in enumerate(data_files):
        try:
            plotDistrib(data, userBins, distributions_to_use[i], plot_file, NBBestFit, ylim[i])
        except Exception as e:
            raise Exception(f"Failed to plot distributions for plotFile{i + 1}: {str(e)}")

    thisTime = time.time()
    logger.debug("Done plotting best distributions in %.2f s", thisTime - startTime)
    logger.info("Process completed")


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
