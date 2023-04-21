###############################################################################################
######################################## MAGE-CNV step 3: Copy numbers calls ##################
###############################################################################################
# Given a TSV of exon fragment counts and a TSV with clustering information,
# obtaining the observation probabilities per copy number (CN), per exon and for each sample.
# See usage for more details.
###############################################################################################
import sys
import getopt
import os
import time
import logging

####### MAGE-CNV modules
import countFrags.countsFile
import clusterSamps.clustFile
import clusterSamps.clustering
import CNCalls.CNCallsFile
import CNCalls.copyNumbersCalls

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
# If anything is wrong, raise Exception("ERROR MESSAGE")
def parseArgs(argv):
    scriptName = os.path.basename(argv[0])

    # mandatory args
    countsFile = ""
    newClustsFile = ""
    outFile = ""
    # optionnal args with default values
    prevCNCallsFile = ""
    prevClustsFile = ""
    plotDir = ""
    checkSamps = ""

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts and a TSV with clustering information,
deduces the copy numbers (CN) observation probabilities, per exon and for each sample.
Results are printed to stdout in TSV format (possibly gzipped): first 4 columns hold the exon
definitions padded and sorted, subsequent columns (four per sample, in order CN0,CN2,CN2,CN3+)
hold the observation probabilities.
If a pre-existing copy number calls file (with --cncalls) produced by this program associated with
a previous clustering file are provided (with --prevclusts), extraction of the observation probabilities
for the samples in homogeneous clusters between the two versions, otherwise the copy number calls is performed.
In addition, all graphical support (pie chart of exon filtering per cluster) are
printed in pdf files created in plotDir.

ARGUMENTS:
    --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns
            hold the fragment counts. File obtained from 1_countFrags.py.
    --clusts [str]: TSV file, contains 5 columns hold the sample cluster definitions.
            [clusterID, sampsInCluster, controlledBy, validCluster, clusterStatus]
            File obtained from 2_clusterSamps.py.
    --out [str]: file where results will be saved, must not pre-exist, will be gzipped if it ends
            with '.gz', can have a path component but the subdir must exist
    --prevcncalls [str] optional: pre-existing copy number calls file produced by this program,
            possibly gzipped, the observation probabilities of copy number types are copied
            for samples contained in immutable clusters between old and new versions of the clustering files.
    --prevclusts [str] optional: pre-existing clustering file produced by s2_clusterSamps.py for the same
            timestamp as the pre-existing copy number call file.
    --plotDir [str]: subdir (created if needed) where result plots files will be produced, wish to monitor the filters
    --checkSamps [str]: comma-separated list of sample names, to plot calls, must be passed with --plotDir
    -h , --help: display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h', ["help", "counts=", "clusts=", "out=", "prevcncalls=",
                                                           "prevclusts=", "plotDir=", "checkSamps="])
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
            newClustsFile = value
        elif opt in ("--out"):
            outFile = value
        elif (opt in ("--prevcncalls")):
            prevCNCallsFile = value
        elif (opt in ("--prevclusts")):
            prevClustsFile = value
        elif (opt in ("--plotDir")):
            plotDir = value
        elif (opt in ("--checkSamps")):
            checkSamps = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check that the mandatory parameter is present
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    if newClustsFile == "":
        raise Exception("you must provide a clustering results file use --clusts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(newClustsFile)):
        raise Exception("clustsFile " + newClustsFile + " doesn't exist.")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other args
    if (prevCNCallsFile != "" and prevClustsFile == "") or (prevCNCallsFile == "" and prevClustsFile != ""):
        raise Exception("you should not use --cncalls and --prevclusts alone but together. Try " + scriptName + " --help")

    if (prevCNCallsFile != "") and (not os.path.isfile(prevCNCallsFile)):
        raise Exception("CNCallsFile " + prevCNCallsFile + " doesn't exist")

    if (prevClustsFile != "") and (not os.path.isfile(prevClustsFile)):
        raise Exception("previous clustering File " + prevClustsFile + " doesn't exist")

    if (plotDir == "" and checkSamps != ""):
        raise Exception("you must use --checkSamps with --plotDir, not independently. Try " + scriptName + " --help")

    # samps2Plot will store user-supplied sample names
    samps2Plot = []
    if plotDir != "":
        # test plotdir last so we don't mkdir unless all other args are OK
        if not os.path.isdir(plotDir):
            try:
                os.mkdir(plotDir)
            except Exception as e:
                raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))
        if checkSamps != "":
            samps2Plot = checkSamps.split(",")

    # AOK, return everything that's needed
    return(countsFile, newClustsFile, outFile, prevCNCallsFile, prevClustsFile, plotDir, samps2Plot)


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
    (countsFile, clustsFile, outFile, prevCNCallsFile, prevClustsFile, plotDir, samps2Plot) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.info("starting to work")
    startTime = time.time()

    ###################
    # parse counts, perform FPM normalization, distinguish between intergenic regions and exons
    try:
        (samples, exons, intergenics, exonsFPM, intergenicsFPM) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.debug("Done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    #####################################################
    # parse clusts
    try:
        (sampsInClusts, ctrlsInClusts, validClusts, specClusts) = clusterSamps.clustFile.parseClustsFile(clustsFile, samples)
    except Exception as e:
        raise Exception("parseClustsFile failed for %s : %s", clustsFile, repr(e))

    thisTime = time.time()
    logger.debug("Done parsing clustsFile, in %.2f s", thisTime - startTime)
    startTime = thisTime

    ######################################################
    # allocate CNcallsArray, and populate it with pre-calculated observed probabilities
    # if CNCallsFile and prevClustFile are provided.
    # also returns a boolean np.array to identify the samples to be reanalysed if the clusters change
    try:
        (CNcallsArray, callsFilled) = CNCalls.CNCallsFile.extractObservedProbsFromPrev(exons, samples, sampsInClusts, prevCNCallsFile, prevClustsFile)
    except Exception as e:
        raise Exception("extractObservedProbsFromPrev failed - " + str(e))

    thisTime = time.time()
    logger.debug("Done parsing previous CNCallsFile and prevClustFile, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # total number of samples that still need to be processed
    nbOfSamplesToProcess = len(samples)
    for samplesIndex in range(len(samples)):
        if callsFilled[samplesIndex]:
            nbOfSamplesToProcess -= 1

    if nbOfSamplesToProcess == 0:
        logger.info("all provided samples are in previous callsFile and clusters are the same, not producing a new one")
    else:
        ####################################################
        # CN Calls
        ##############################
        # identifying autosomes and gonosomes "exons" index
        # recall clusters are derived from autosomal or gonosomal analyses
        maskGExIndexes = clusterSamps.getGonosomesExonsIndexes.getSexChrIndexes(exons)
        
        for clustID in range(len(sampsInClusts)):
            # creation of folder for storing monitoring plots
            if plotDir:
                pathDirPlotCN = CNCalls.copyNumbersCalls.makePlotDir(plotDir, clustID)
            else:
                pathDirPlotCN = ""
                samps2Plot = ""

            ##### validity sanity check
            if validClusts[clustID] == 0:
                logger.warning("cluster %s is invalid, low sample number", clustID)
                continue

            ##### previous data sanity filters
            # test if the samples of the cluster have already been analysed
            # and the filling of CNcallsArray has been done
            sub_t = callsFilled[sampsInClusts[clustID]]
            if all(sub_t):
                logger.info("samples in cluster %s, already filled from prevCallsFile", clustID)
                continue

            # get cluster-specific data, all samples in the case of a presence of a control cluster,
            # and exon indexes that need to be analyzed
            (allSampsInClust, exIndToProcess) = CNCalls.copyNumbersCalls.getSampsAndEx2Process(clustID, sampsInClusts, ctrlsInClusts, specClusts, maskGExIndexes)

            try:
                CNCalls.copyNumbersCalls.CNCalls(CNcallsArray, clustID, exonsFPM, intergenicsFPM, samples, allSampsInClust,
                                                 sampsInClusts[clustID], exons, exIndToProcess, pathDirPlotCN, samps2Plot)
            except Exception as e:
                raise Exception("CNCalls failed %s", e)

            thisTime = time.time()
            logger.debug("Done Copy Number Calls, in %.2f s", thisTime - startTime)
            startTime = thisTime

        #####################################################
        # Print exon defs + calls to outFile
        CNCalls.CNCallsFile.printCNCallsFile(CNcallsArray, exons, samples, outFile)

        thisTime = time.time()
        logger.debug("Done printing calls for all (non-failed) samples, in %.2fs", thisTime - startTime)
        logger.info("ALL DONE")


####################################################################################
######################################## Main ######################################
####################################################################################
if __name__ == '__main__':
    # configure logging, sub-modules will inherit this config
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    # set up logger: we want script name rather than 'root'
    logger = logging.getLogger(os.path.basename(sys.argv[0]))

    try:
        main(sys.argv)
    except Exception as e:
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + sys.argv[0] + " : " + str(e) + "\n")
        sys.exit(1)
