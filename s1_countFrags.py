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
######################################## JACNEx step 1: count reads ###########################
###############################################################################################
# Given a BED of exons and one or more BAM files, count the number of sequenced fragments
# from each BAM that overlap each exon (+- padding).
# See usage for details.
###############################################################################################
import concurrent.futures
import getopt
import gzip
import logging
import math
import numpy
import os
import re
import shutil
import sys
import time

####### JACNEx modules
import countFrags.bed
import countFrags.countsFile
import countFrags.countFragments


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
    bams = ""
    bamsFrom = ""
    bedFile = ""
    outFile = ""
    # optional args with default values
    BPDir = "BreakPoints/"
    countsFile = ""
    padding = 10
    maxGap = 1000
    tmpDir = "/tmp/"
    samtools = "samtools"
    # jobs default: 80% of available cores
    jobs = round(0.8 * len(os.sched_getaffinity(0)))

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a BED of exons and one or more BAM files:
- pad the exons (+- padding), merge overlapping padded exons, and create intergenic (or deep
    intronic) pseudo-exons in the larger gaps between exons;
- count the number of sequenced fragments from each BAM that overlap each (pseudo-)exon.
Results (sample names, genomic windows, counts) are saved to --out in NPZ format.
The "sample names" are the BAM filenames stripped of their path and .bam extension, they
are sorted alphanumerically.
In addition, any support for putative breakpoints is printed to sample-specific TSV.gz files
created in BPDir.
If a pre-existing counts file produced by this program with the same BED is provided (with --counts),
counts for common BAMs are copied from this file and counting is only performed for the new BAMs.
Furthermore, if the BAMs exactly match those in --counts, the output file (--out) is not produced.

ARGUMENTS:
   --bams [str] : comma-separated list of BAM files (with path)
   --bams-from [str] : text file listing BAM files (with path), one per line
   --bed [str] : BED file, possibly gzipped, containing exon definitions (format: 4-column
           headerless tab-separated file, columns contain CHR START END EXON_ID)
   --out [str] : file where results will be saved (unless BAMs exactly match those in --counts),
           must not pre-exist, can have a path component but the subdir must exist
   --BPDir [str] : dir (created if needed) where breakpoint files will be produced, default :  """ + BPDir + """
   --counts [str] optional: pre-existing counts file produced by this program
   --jobs [int] : cores that we can use, defaults to 80% of available cores ie """ + str(jobs) + "\n" + """
   --padding [int] : number of bps used to pad the exon coordinates, default : """ + str(padding) + """
   --maxGap [int] : maximum accepted gap length (bp) between reads pairs, pairs separated by a longer gap
           are assumed to possibly result from a structural variant and are ignored, default : """ + str(maxGap) + """
   --tmp [str] : pre-existing dir for temp files, faster is better (eg tmpfs), default: """ + tmpDir + """
   --samtools [str] : samtools binary (with path if not in $PATH), default: """ + samtools + """
   -h , --help : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "bams=", "bams-from=", "bed=", "out=", "BPDir=",
                                                       "counts=", "jobs=", "padding=", "maxGap=", "tmp=", "samtools="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--bams"):
            bams = value
        elif opt in ("--bams-from"):
            bamsFrom = value
        elif opt in ("--bed"):
            bedFile = value
        elif opt in ("--out"):
            outFile = value
        elif opt in ("--BPDir"):
            BPDir = value
        elif opt in ("--counts"):
            countsFile = value
        elif opt in ("--jobs"):
            jobs = value
        elif opt in ("--padding"):
            padding = value
        elif opt in ("--maxGap"):
            maxGap = value
        elif opt in ("--tmp"):
            tmpDir = value
        elif opt in ("--samtools"):
            samtools = value
        else:
            raise Exception("unhandled option " + opt)

    #####################################################
    # Check, clean up and sort list of BAMs
    if (bams == "" and bamsFrom == "") or (bams != "" and bamsFrom != ""):
        raise Exception("you must use either --bams or --bams-from but not both. Try " + scriptName + " --help")
    # bamsTmp will store user-supplied BAMs
    bamsTmp = []
    # lists of BAMs and samples to process, where sample names are the BAM name
    # stripped of path and .bam extension, both will be sorted by sample name
    bamsToProcess = []
    samples = []
    # samplesSeen: tmp dictionary for identifying dupes if any: key==sample, value==1
    samplesSeen = {}

    if bams != "":
        bamsTmp = bams.split(",")
    elif not os.path.isfile(bamsFrom):
        raise Exception("bams-from file " + bamsFrom + " doesn't exist")
    else:
        try:
            bamsList = open(bamsFrom, "r")
            for bam in bamsList:
                bam = bam.rstrip()
                bamsTmp.append(bam)
            bamsList.close()
        except Exception as e:
            raise Exception("opening provided --bams-from file " + bamsFrom + " : " + str(e))

    # Check that all bams exist and that there aren't any duplicate samples
    for bam in bamsTmp:
        if not os.path.isfile(bam):
            raise Exception("BAM " + bam + " doesn't exist")
        elif not os.access(bam, os.R_OK):
            raise Exception("BAM " + bam + " cannot be read")
        else:
            sample = os.path.basename(bam)
            sample = re.sub(r"\.[bs]am$", "", sample)
            if sample in samplesSeen:
                logger.error("multiple BAMs correspond to sample " + sample + ", this is not allowed")
                raise Exception("multiple BAMs for sample " + sample)
            samplesSeen[sample] = 1
            bamsToProcess.append(bam)
            samples.append(sample)
    # sort both lists by sampleName
    sampleIndexes = list(range(len(samples)))
    sampleIndexes.sort(key=samples.__getitem__)
    samples = [samples[i] for i in sampleIndexes]
    bamsToProcess = [bamsToProcess[i] for i in sampleIndexes]

    #####################################################
    # Check other args
    if bedFile == "":
        raise Exception("you must provide a BED file with --bed. Try " + scriptName + " --help")
    elif not os.path.isfile(bedFile):
        raise Exception("bedFile " + bedFile + " doesn't exist")

    if outFile == "":
        raise Exception("you must provide an outFile with --out. Try " + scriptName + " --help")
    elif os.path.exists(outFile):
        raise Exception("outFile " + outFile + " already exists")
    elif (os.path.dirname(outFile) != '') and (not os.path.isdir(os.path.dirname(outFile))):
        raise Exception("the directory where outFile " + outFile + " should be created doesn't exist")

    if (countsFile != "") and (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist")

    try:
        jobs = int(jobs)
        if (jobs <= 0):
            raise Exception()
    except Exception:
        raise Exception("jobs must be a positive integer, not " + str(jobs))

    try:
        padding = int(padding)
        if (padding < 0):
            raise Exception()
    except Exception:
        raise Exception("padding must be a non-negative integer, not " + str(padding))

    try:
        maxGap = int(maxGap)
        if (maxGap < 0):
            raise Exception()
    except Exception:
        raise Exception("maxGap must be a non-negative integer, not " + str(maxGap))

    if not os.path.isdir(tmpDir):
        raise Exception("tmp directory " + tmpDir + " doesn't exist")

    if shutil.which(samtools) is None:
        raise Exception("samtools program '" + samtools + "' cannot be run (wrong path, or binary not in $PATH?)")

    # test BPDir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(BPDir):
        try:
            os.mkdir(BPDir)
        except Exception as e:
            raise Exception("BPDir " + BPDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(bamsToProcess, samples, bedFile, outFile, BPDir, jobs, padding, maxGap, countsFile, tmpDir, samtools)


####################################################
# calcParaSamples: decide how many samples to process in parallel.
# We are allowed to use jobs cores in total: we will process paraSamples samples in
# parallel, each sample will be processed using coresPerSample.
#
# Return 2 ints: (paraSamples, coresPerSample)
def calcParaSamples(jobs, bamsToProcess, countsFilled, nbOfSamplesToProcess, tmpDir):
    # In our tests 7 cores/sample provided the best overall performance for WES data, both
    # with jobs=20 and jobs=40 on a dual-CPU Xeon Silver 4114.
    # This is surely hardware-dependant, but it's just a performance tuning parameter.
    # However, more samples in parallel require more RAM and more space on tmpDir...
    # overfilling the RAM or tmpDir will result in a crash.
    # -> we target targetCoresPerSample coresPerSample to calculate paraSamples
    targetCoresPerSample = 7
    paraSamples = math.ceil(jobs / targetCoresPerSample)
    # never more paraSamples than the total number of samples
    paraSamples = min(paraSamples, nbOfSamplesToProcess)

    # don't fill up tmpDir: samtools may need as much space as the paraSamples largest BAMs
    (totalTmp, usedTmp, freeTmp) = shutil.disk_usage(tmpDir)
    # we will limit tmpDir usage to max 90%
    freeTmp *= 0.9
    bamSizes = []
    for bamIndex in range(len(bamsToProcess)):
        if countsFilled[bamIndex]:
            continue
        bam = bamsToProcess[bamIndex]
        bamSizes.append(os.path.getsize(bam))
    bamSizes.sort(reverse=True)
    maxSamples = 0
    sumOfSizes = 0
    while (maxSamples < paraSamples):
        sumOfSizes += bamSizes[maxSamples]
        if (sumOfSizes > freeTmp):
            break
        else:
            maxSamples += 1
    if (maxSamples == 0):
        logger.error("not enough space on tmpDir %s, cannot process the largest BAM (even alone)", tmpDir)
        raise Exception("not enough space on tmpDir")
    if (maxSamples < paraSamples):
        logger.info("limiting samples processed in parallel due to low tmpDir space, increase space if possible")
        paraSamples = maxSamples

    # unsure how much RAM is needed for each sample, in our hands we don't have RAM issues.
    # If OOM crashes occur: open an issue on github, we will need to estimate the RAM usage
    # per sample, and limit paraSamples accordingly

    # we use ceil() so we may slighty overconsume
    coresPerSample = math.ceil(jobs / paraSamples)
    return(paraSamples, coresPerSample)


####################################################
# main function
# Arg: list of strings, eg sys.argv
# If anything goes wrong, raise Exception("SOME EXPLICIT ERROR MESSAGE"), more details
# may be available in the log
def main(argv):
    # parse, check and preprocess arguments
    (bamsToProcess, samples, bedFile, outFile, BPDir, jobs, padding, maxGap, countsFile, tmpDir, samtools) = parseArgs(argv)

    # args seem OK, start working
    logger.debug("called with: " + " ".join(argv[1:]))
    logger.debug("starting to work")
    startTime = time.time()

    # parse exons from BED to obtain a list of lists (dim=(NbMergedExons+NbPseudoExons) x [CHR,START,END,EXONID]),
    # the exons and pseudo-exons are sorted according to their genomic position and padded
    exons = countFrags.bed.processBed(bedFile, padding)

    thisTime = time.time()
    logger.info("Done pre-processing BED, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # allocate countsArray and countsFilled, and populate them with pre-calculated
    # counts if countsFile was provided.
    # countsArray[exonIndex,sampleIndex] will store the specified count,
    # countsFilled[sampleIndex] is True iff counts for specified sample were filled from countsFile
    try:
        (countsArray, countsFilled) = countFrags.countsFile.extractCountsFromPrev(exons, samples, countsFile)
    except Exception as e:
        raise Exception("extractCountsFromPrev failed - " + str(e))

    thisTime = time.time()
    logger.info("Done parsing previous countsFile, in %.2fs", thisTime - startTime)
    startTime = thisTime

    # total number of samples that still need to be processed
    nbOfSamplesToProcess = len(bamsToProcess)
    for bamIndex in range(len(bamsToProcess)):
        if countsFilled[bamIndex]:
            nbOfSamplesToProcess -= 1

    # if bam2counts fails for any BAMs, we have to remember their indexes
    # and only expunge them at the end -> save their indexes in failedBams
    failedBams = []

    if nbOfSamplesToProcess == 0:
        logger.info("all provided BAMs are in previous countsFile")
        # if samples exactly match those in countsFile, return immediately
        prevSamples = countFrags.countsFile.countsFile2samples(countsFile)
        if prevSamples == samples:
            logger.info("provided BAMs exactly match those in previous countsFile, not producing a new one")
            return()

    else:
        #####################################################
        # decide how the work will be parallelized
        (paraSamples, coresPerSample) = calcParaSamples(jobs, bamsToProcess, countsFilled,
                                                        nbOfSamplesToProcess, tmpDir)
        logger.info("%i new sample(s)  => will process %i in parallel, using up to %i cores/sample",
                    nbOfSamplesToProcess, paraSamples, coresPerSample)

        #####################################################
        # populate the module-global exonNCLs in countFragments
        try:
            countFrags.countFragments.initExonNCLs(exons)
        except Exception as e:
            raise Exception("initExonNCLs failed - " + str(e))

        #####################################################
        # Define nested callback for processing bam2counts() result (so countsArray et al
        # are in its scope)

        # mergeCounts:
        # arg: a Future object returned by ProcessPoolExecutor.submit(countFrags.countFragments.bam2counts).
        # bam2counts() returns a 3-element tuple (sampleIndex, sampleCounts, breakPoints).
        # If something went wrong, log and populate failedBams;
        # otherwise fill column at index sampleIndex in countsArray with counts stored in sampleCounts,
        # and print info about putative CNVs with alignment-supported breakpoints as TSV.gz
        # to BPDir/<sample>.breakPoints.tsv.gz
        def mergeCounts(futureBam2countsRes):
            e = futureBam2countsRes.exception()
            if e is not None:
                #  exceptions raised by bam2counts are always Exception(str(sampleIndex))
                si = int(str(e))
                logger.warning("Failed to count fragments for sample %s, skipping it", samples[si])
                failedBams.append(si)
            else:
                bam2countsRes = futureBam2countsRes.result()
                si = bam2countsRes[0]
                countsArray[:, si] = bam2countsRes[1][:]
                if (len(bam2countsRes[2]) > 0):
                    try:
                        # NOTE: keep filename in sync with bpFile in callsFile.py:parseBreakpoints()
                        bpFile = os.path.join(BPDir, samples[si] + '.breakPoints.tsv.gz')
                        BPFH = gzip.open(bpFile, "xt", compresslevel=6)
                        BPFH.write(bam2countsRes[2])
                        BPFH.close()
                    except Exception as e:
                        logger.warning("Discarding breakpoints info for %s because cannot gzip-open %s for writing - %s",
                                       samples[si], bpFile, e)
                logger.info("Done counting fragments for %s", samples[si])

        #####################################################
        # Process new BAMs, up to paraSamples in parallel
        with concurrent.futures.ProcessPoolExecutor(paraSamples) as pool:
            for bamIndex in range(len(bamsToProcess)):
                bam = bamsToProcess[bamIndex]
                if countsFilled[bamIndex]:
                    # logger.debug('Sample %s already filled from countsFile', samples[bamIndex])
                    continue
                else:
                    futureRes = pool.submit(countFrags.countFragments.bam2counts,
                                            bam, len(exons), maxGap, tmpDir, samtools, coresPerSample, bamIndex)
                    futureRes.add_done_callback(mergeCounts)

        #####################################################
        # Expunge samples for which bam2counts failed
        if len(failedBams) > 0:
            for failedI in reversed(failedBams):
                del(samples[failedI])
            countsArray = numpy.delete(countsArray, failedBams, axis=1)

        thisTime = time.time()
        logger.info("Processed %i new BAMs in %.2fs, i.e. %.2fs per BAM",
                    nbOfSamplesToProcess, thisTime - startTime, (thisTime - startTime) / nbOfSamplesToProcess)
        startTime = thisTime

    #####################################################
    # save exon defs + samples + counts to outFile
    countFrags.countsFile.printCountsFile(exons, samples, countsArray, outFile)

    thisTime = time.time()
    logger.info("Done saving counts for all (non-failed) samples, in %.2fs", thisTime - startTime)
    if len(failedBams) > 0:
        raise("counting FAILED for " + len(failedBams) + " samples, check the log!")
    else:
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
        # details on the issue should be in the exception name, print it to stderr and die
        sys.stderr.write("ERROR in " + scriptName + " : " + str(e) + "\n")
        sys.exit(1)
