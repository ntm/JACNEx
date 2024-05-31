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
######################################## JACNEx step 2: Sample clustering  ####################
###############################################################################################
# Given a TSV of fragment counts as produced by 1_countFrags.py:
# build clusters of samples that will be used as controls for one another.
# See usage for details.
###############################################################################################
import getopt
import logging
import os
import sys
import time

####### JACNEx modules
import clusterSamps.clustering
import clusterSamps.clustFile
import clusterSamps.gender
import countFrags.bed
import countFrags.countsFile

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
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
    outFile = ""
    # optional args with default values
    minSamps = 20
    plotDir = "./plotDir/"

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a TSV of exon fragment counts, build clusters of "similar" samples that
will be used as controls for one another.
Clusters are built independantly for exons on autosomes ('A') and  on gonosomes ('G').
The accepted sex chromosomes are X, Y, Z, and W.
Results are printed to --out in TSV format: 5 columns
[CLUSTER_ID, FIT_WITH, GENDER, VALID, SAMPLES]
In addition, dendrograms of the clustering results are produced as pdf files in plotDir.

ARGUMENTS:
   --counts [str]: TSV file of fragment counts, possibly gzipped, produced by s1_countFrags.py
   --out [str] : file where clusters will be saved, must not pre-exist, will be gzipped if it ends
                 with '.gz', can have a path component but the subdir must exist
   --minSamps [int]: minimum number of samples for a cluster to be declared valid, default : """ + str(minSamps) + """
   --plotDir [str]: subdir (created if needed) where plot files will be produced, default:  """ + plotDir + """
   -h , --help  : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "out=", "minSamps=", "plotDir="])
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
        elif opt in ("--out"):
            outFile = value
        elif (opt in ("--minSamps")):
            minSamps = value
        elif (opt in ("--plotDir")):
            plotDir = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check mandatory args
    if countsFile == "":
        raise Exception("you must provide a countsFile with --counts. Try " + scriptName + " --help")
    elif not os.path.isfile(countsFile):
        raise Exception("countsFile " + countsFile + " doesn't exist")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    #####################################################
    # Check other args
    try:
        minSamps = int(minSamps)
        if (minSamps <= 1):
            raise Exception()
    except Exception:
        raise Exception("minSamps must be an integer > 1, not " + str(minSamps))

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, outFile, minSamps, plotDir)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (countsFile, outFile, minSamps, plotDir) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.debug("starting to work")
    startTime = time.time()

    ###################
    # parse and FPM-normalize the counts, distinguishing between exons and intergenic pseudo-exons
    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, repr(e))
        raise Exception("parseAndNormalizeCounts failed")

    thisTime = time.time()
    logger.info("done parseAndNormalizeCounts, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # cannot make any useful clusters if too few samples, in fact sklearn.decomposition
    # (PCA) dies with an ugly message if called with a single sample
    if len(samples) < minSamps:
        logger.error("JACNEx requires at least minSamps (%i) samples, you provided %i, aborting",
                     minSamps, len(samples))
        raise Exception("JACNEx needs more samples to go beyond counting")

    ###################
    # Clustering:
    # build clusters of samples with "similar" count profiles, independently for exons
    # located on autosomes and on sex chromosomes (==gonosomes).
    # As a side benefit this also allows to identify same-gender samples, since samples
    # that cluster together for the gonosomal exons always have the same gonosomal
    # karyotype (predominantly on the X: in our hands XXY samples always cluster with XX
    # samples, not with XY ones)

    # build root name for dendrograms, will just need to append autosomes.pdf or gonosomes.pdf
    dendroFileRoot = os.path.basename(outFile)
    # remove file extension (.tsv probably), and also .gz if present
    if dendroFileRoot.endswith(".gz"):
        dendroFileRoot = os.path.splitext(dendroFileRoot)[0]
    dendroFileRoot = os.path.splitext(dendroFileRoot)[0]
    dendroFileRoot = dendroFileRoot + "_dendrogram"
    dendroFileRoot = os.path.join(plotDir, dendroFileRoot)

    # autosomes
    try:
        plotFile = dendroFileRoot + "_autosomes.pdf"
        (clust2samps, fitWith, clustIsValid) = clusterSamps.clustering.buildClusters(
            autosomeFPMs, "A", samples, minSamps, plotFile)
    except Exception as e:
        logger.error("buildClusters failed for autosomes: %s", repr(e))
        raise Exception("buildClusters failed")

    thisTime = time.time()
    logger.info("done clustering samples for autosomes, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # sex chromosomes
    try:
        plotFile = dendroFileRoot + "_gonosomes.pdf"
        (clust2sampsGono, fitWithGono, clustIsValidGono) = clusterSamps.clustering.buildClusters(
            gonosomeFPMs, "G", samples, minSamps, plotFile)
    except Exception as e:
        logger.error("buildClusters failed for gonosomes: %s", repr(e))
        raise Exception("buildClusters failed")
    clust2samps.update(clust2sampsGono)
    fitWith.update(fitWithGono)
    clustIsValid.update(clustIsValidGono)

    thisTime = time.time()
    logger.info("done clustering samples for gonosomes, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # predict genders
    clust2gender = clusterSamps.gender.assignGender(
        gonosomeFPMs, intergenicFPMs, gonosomeExons, samples, clust2sampsGono, fitWithGono)

    thisTime = time.time()
    logger.info("done predicting genders, in %.2fs", thisTime - startTime)
    startTime = thisTime

    ###################
    # print clustering results
    try:
        clusterSamps.clustFile.printClustsFile(clust2samps, fitWith, clust2gender, clustIsValid, outFile)
    except Exception as e:
        logger.error("printing clusters failed : %s", repr(e))
        raise Exception("printClustsFile failed")

    logger.debug("ALL DONE")


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
        sys.stderr.write("ERROR in " + scriptName + " : " + repr(e) + "\n")
        sys.exit(1)
