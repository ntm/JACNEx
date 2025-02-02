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


###############################################################################################
# Given a clusterFile produced by s2_clusterSamps.py and a metadata file in TSV format,
# print to stdout the same clusterFile but annotated with additional columns.
# See usage for details.
###############################################################################################
import getopt
import logging
import os
import sys
# import traceback

####### JACNEx modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import clusterSamps.clustFile  # noqa: E402

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
    clusterFile = ""
    metadataFile = ""

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a clusterFile produced by s2_clusterSamps.py and a metadata file in TSV format:
- define "batches" of samples that should be similar given the metdata. This is
    mostly a combination of the sequencing center, capture kit and date (when it
    should be relevant, based on a review of all our currently available exomes).
    For gonosome clusters it includes the gender/karyoype: "M", "F", or "XXY".
    The metadata columns & content are our usual (as in grexome-TIMC pipelines)
    but in TSV format, see parseMetadata() for the required columns.
- print to stdout the input clusterFile but annotated with additional columns:
    NUMBER_Of_BATCHES: the number of different batches that this cluster's samples
    belong to;
    BATCHES: the list of these batches, along with the number of samples from this
    cluster that belong to each one;
    NUMBER_Of_BATCHES_FITWITH, BATCHES_FITWITH: similar but also includes the
    FIT_WITH clusters.

ARGUMENTS:
   --clusters [str] : input clusterFile (with path)
   --metadata [str] : metadata file in TSV format (with path)
   -h , --help : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "clusters=", "metadata="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--clusters"):
            clusterFile = value
        elif opt in ("--metadata"):
            metadataFile = value
        else:
            raise Exception("unhandled option " + opt)

    # Check args
    if clusterFile == "":
        raise Exception("you must provide a clusterFile file with --clusters. Try " + scriptName + " --help")
    elif not os.path.isfile(clusterFile):
        raise Exception("clusterFile " + clusterFile + " doesn't exist")

    if metadataFile == "":
        raise Exception("you must provide a metadata file with --metadata. Try " + scriptName + " --help")
    elif not os.path.isfile(metadataFile):
        raise Exception("metadataFile " + metadataFile + " doesn't exist")

    # AOK, return everything that's needed
    return(clusterFile, metadataFile)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# Prints annotated clusterFile to STDOUT
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    (clusterFile, metadataFile) = parseArgs(argv)

    try:
        (samp2sex, samp2batch) = parseMetadata(metadataFile)
    except Exception as e:
        logger.error("error parsing metadata file: %s", str(e))
        raise
    try:
        (clust2samps, samp2clusts, fitWith, clust2gender, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clusterFile)
    except Exception as e:
        logger.error("error parsing clusterFile: %s", str(e))
        raise

    toPrint = "CLUSTER_ID\tFIT_WITH\tGENDER\tVALID\tSAMPLES"
    # new columns, see USAGE for their content
    toPrint += "\tNUMBER_Of_BATCHES\tBATCHES\tNUMBER_Of_BATCHES_FITWITH\tBATCHES_FITWITH"
    print(toPrint)

    # clust2batches: key is a clusterID, value is a dict with key == batchID (samp2batch
    # for autosomes and samp2sex_samp2batch for gonosomes), and value == the number of
    # samples belonging to this batch in this cluster
    clust2batches = {}
    # clust2batchesFW: same but includes samples belonging to FITWITH clusters
    clust2batchesFW = {}

    for clusterID in clust2samps.keys():
        # batches: key is a batchID, value is the number of samples belonging to
        # this batch in this cluster
        batches = {}
        for samp in clust2samps[clusterID]:
            batchID = ""
            if clusterID.startswith("A_"):
                batchID = samp2batch[samp]
            else:
                batchID = samp2sex[samp] + "_" + samp2batch[samp]
            if batchID in batches:
                batches[batchID] += 1
            else:
                batches[batchID] = 1
        clust2batches[clusterID] = batches

    for clusterID in clust2samps.keys():
        # batchesFitWith: same as batches above but also includes samples from FIT_WITH clusters
        batchesFitWith = {}
        batchesFitWith = clust2batches[clusterID].copy()
        for fw in fitWith[clusterID]:
            for batchFW in clust2batches[fw].keys():
                if batchFW in batchesFitWith:
                    batchesFitWith[batchFW] += clust2batches[fw][batchFW]
                else:
                    batchesFitWith[batchFW] = clust2batches[fw][batchFW]
        clust2batchesFW[clusterID] = batchesFitWith

    for clusterID in sorted(clust2samps.keys()):
        toPrint = clusterID + "\t"
        toPrint += ','.join(sorted(fitWith[clusterID])) + "\t"
        if clusterID in clust2gender:
            toPrint += clust2gender[clusterID] + "\t"
        else:
            toPrint += "\t"
        if clustIsValid[clusterID]:
            toPrint += "1\t"
        else:
            toPrint += "0\t"
        toPrint += ','.join(sorted(clust2samps[clusterID])) + "\t"

        toPrint += str(len(clust2batches[clusterID].keys())) + "\t"
        for b in sorted(clust2batches[clusterID].keys()):
            toPrint += b + ":" + str(clust2batches[clusterID][b]) + ","
        toPrint += "\t"

        toPrint += str(len(clust2batchesFW[clusterID].keys())) + "\t"
        for b in sorted(clust2batchesFW[clusterID].keys()):
            toPrint += b + ":" + str(clust2batchesFW[clusterID][b]) + ","

        print(toPrint)


##############################################
# parseMetadata:
# Args:
#   -metadataFile is a TSV file with at least these columns, in any order:
#        "sampleID", empty or "0" are ignored
#        "Sex"
#        "Quality control"
#        "Center"
#        "capture"
#        "Date"
# Returns:
# samp2sex: key == sampleID, value == "M", "F" (including for the 2 "suspected XX"),
#      or "XXY" for the 4 "suspected XXY / contamination" samples
# samp2batch: key == sampleID, value is a string that identifies the batch this sample
#      should belong to according to metadata ie "Center_captureKit" for everyone except
#      Genoscope and Novogene: Center_captureKit_Date
#
# NOTE the batch identifiers are based on my analysis of all our samples as of today, counts are:
# 201 Novogene 10/2017 (2 "suspected XX")
# 19 Novogene 08/2016 (same kit but diff sequencer vs Novo 2017)
# 23 Novogene 05/2019 (same kit but diff sequencer vs both prev Novo, 1 "XXY / contamination")
# 12 Novogene 03/2020 (same kit & seq as Novo 2019, 3 "XXY / contamination")
# 40 Integragen (grexomes 433-436 + a series in small grexome numbers)
# 29 Genoscope 11/2013
# 76 Genoscope 10/2014 (same kit & seq as prev Genoscope)
# 7 Genoscope 02/2015 (same seq & kit as prev Genoscopes)
# 12 Strasbourg
# 6 Versailles 04/2017
# 23 Versailles 03/2018 (diff kit vs Versailles 2017)
# 5 Versailles 10/2018 (same kit & seq as Versailles 2017 BUT NOT 03/2018)!
# 25 BGI
# 17 Macrogen 06/2021
# 1 Macrogen 08/2023 (solo sample, twist)
# 40 Imagine
# 15 Biomnis: 7 + 2 + 6 (continuous dates but a few diff kits)
# 59 IBP illumina_Truseq_Exome_v1.2 (continuous dates)
# 68 IBP Agilent_v8 (continuous dates)
def parseMetadata(metadataFile):
    try:
        metadataFH = open(metadataFile, "r")
    except Exception as e:
        logger.error("Opening provided CSV metadata file %s: %s", metadataFile, e)
        raise Exception("cannot open provided metadata file")

    ######################
    # parse header
    headers = metadataFH.readline().rstrip().split("\t")

    samp2sex = {}
    samp2batch = {}

    for line in metadataFH:
        splitLine = line.rstrip().split("\t")
        sample = splitLine[headers.index("sampleID")]
        if (sample == "") or (sample == '0'):
            continue

        sex = splitLine[headers.index("Sex")]
        qc = splitLine[headers.index("Quality control")]
        if qc == "suspected XX":
            sex = 'F'
        elif (qc == "suspected XXY") or (qc == "suspected XXY or contamination"):
            sex = 'XXY'
        elif qc == "suspected contamination":
            pass  # == noop
        elif qc != '':
            logger.error("parsing QC column in metadata file %s, found unexpected/unimplemented value %s",
                         metadataFile, qc)
            raise Exception("unimplemented QC value in metadata file")

        samp2sex[sample] = sex

        batch = splitLine[headers.index("Center")] + "_" + splitLine[headers.index("capture")]
        if batch.startswith('Genoscope') or batch.startswith('Novogene'):
            batch += "_" + splitLine[headers.index("Date")]
        samp2batch[sample] = batch

    return(samp2sex, samp2batch)


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
        # traceback.print_exc()
        sys.exit(1)
