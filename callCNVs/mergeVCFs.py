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


import gzip
import hashlib
import logging
import re

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
##########################################
# mergeVCFs:
# given a list of samples and a list of VCF files corresponding to different
# clusters (each concerning a subset of the samples), produce a single VCF
# file with data columns in the "samples" order, one variant per line:
# -> identical chrom-pos-ALT-END from different files get merged;
# -> if chrom-pos-ALT-END is not present in a file, samples from this file get 0/0.
#
# NOTE this strategy isn't perfect: it pretends that all clusters examined
# the same exons, but in reality some exons may be CALL in some clusters and
# NOCALL in others (eg different capture kits)... A better strategy could be to
# set 0/0 to samples whose cluster has at least one CALLED exon overlapping the
# CNV, and ./. otherwise. However this would require recording (in the VCF?) the
# called exons for each cluster, because JACNEx reuses pre-existing VCFs for
# clusters whose samples didn't change since the previous run... so when merging
# we don't currently know the called-but-all-samples-homoref exons.
# Actually, even within a single cluster, printCallsFile() is already pretending
# that samples are 0/0 (rather than ./.) when they are non-HR for an overlapping CNV...
# For now we'll just use 0/0.
#
# Args:
# - samples: list of strings == sampleIDs
# - infiles: list of strings, each string is a VCF filename (with path) produced
#   by JACNEx for one cluster
# - outFile (str): name of VCF file to create. Will be squashed if pre-exists,
#   can include a path (which must exist), will be gzipped if outFile ends with '.gz'
def mergeVCFs(samples, inFiles, outFile):
    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open merged VCF file %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open merged VCF')

    # inFH: key=inFile, value= filehandle open for reading
    inFH = {}
    try:
        for inFile in inFiles:
            if inFile.endswith(".gz"):
                inFH[inFile] = gzip.open(inFile, "rt")
            else:
                inFH[inFile] = open(inFile, "r")
    except Exception as e:
        logger.error("in mergeVCFs, cannot open inFile %s: %s", inFile, e)
        raise Exception('mergeVCFs cannot open inFile')

    # samp2i: key==sampleID, value==index of sampleID in samples
    samp2i = {}
    for si in range(len(samples)):
        samp2i[samples[si]] = si

    # headers: copy from first VCF, discard headers from other VCFs but
    # build a mapping: for each inFile, clust2global[inFile][i] = j means that
    # sample in data column i in inFile is the sample in data column j in outFile
    clust2global = {}
    toPrint = ""
    for ifi in range(len(inFiles)):
        thisFile = inFiles[ifi]
        for line in inFH[thisFile]:
            if line.startswith('#CHROM'):
                fields = line.rstrip().split("\t")
                # first 9 fields are #CHROM...FORMAT (VCF spec)
                if (ifi == 0):
                    for i in range(9):
                        toPrint += fields[i] + "\t"
                    toPrint += "\t".join(samples) + "\n"
                del fields[:9]
                clust2global[thisFile] = [None] * len(fields)
                for i in range(len(fields)):
                    clust2global[thisFile][i] = samp2i[fields[i]]
                break
            elif (ifi == 0):
                toPrint += line
            # else this is a non-#CHROM header line and not the first file, ignore
    outFH.write(toPrint)

    # data:
    #  nextLines is a list of "lines", one for each inFile - a "line" here is
    # a list of strings (ie tab-split the actual VCF line), or False if
    # inFile has no more lines
    nextLines = []
    # also maintain a synchronized list of [chromAsInt, Pos, End, SVType] of types
    # [int, int, int, string] to ease comparisons
    nextLinesForComp = []
    for thisFile in inFiles:
        line = inFH[thisFile].readline()
        if (line):
            fields = line.rstrip().split("\t")
            nextLines.append(fields)
            nextLinesForComp.append(line2sortable(fields))
        else:
            nextLines.append(False)
            nextLinesForComp.append(False)

    while(any(nextLines)):
        # find indexes of "smallest" (sort order ==chrom,pos,end,svType) nextLines
        smallest = []
        for ifi in range(len(inFiles)):
            if nextLinesForComp[ifi]:
                if (len(smallest) == 0) or (nextLinesForComp[ifi] == nextLinesForComp[smallest[0]]):
                    smallest.append(ifi)
                else:
                    prev = nextLinesForComp[smallest[0]]
                    new = nextLinesForComp[ifi]
                    if (new[0] < prev[0]):
                        smallest = [ifi]
                    elif (new[0] == prev[0]):
                        if (new[1] < prev[1]):
                            smallest = [ifi]
                        elif (new[1] == prev[1]):
                            if (new[2] < prev[2]):
                                smallest = [ifi]
                            elif (new[2] == prev[2]):
                                if (new[3] < prev[3]):
                                    smallest = [ifi]
                                # else new[3] must be > prev[3] since equality was already tested
        # create full VCF line from smallest line(s)
        startOfLine = nextLines[smallest[0]][:9]
        del nextLines[smallest[0]][:9]
        # if FORMAT has BP (optional) in any smallest line, use it
        for smi in range(1, len(smallest)):
            if (len(startOfLine[8]) < len(nextLines[smallest[smi]][8])):
                startOfLine[8] = nextLines[smallest[smi]][8]
            del nextLines[smallest[smi]][:9]
        # data: default to HOMOREF for all samples
        data = ['0/0'] * len(samples)
        for sm in smallest:
            for i in range(len(nextLines[sm])):
                data[clust2global[inFiles[sm]][i]] = nextLines[sm][i]
        toPrint = "\t".join(startOfLine) + "\t" + "\t".join(data) + "\n"
        outFH.write(toPrint)
        # grab next lines
        for sm in smallest:
            line = inFH[inFiles[sm]].readline()
            if (line):
                fields = line.rstrip().split("\t")
                nextLines[sm] = fields
                nextLinesForComp[sm] = line2sortable(fields)
            else:
                nextLines[sm] = False
                nextLinesForComp[sm] = False

    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# line2sortable:
# given a VCF "line" == a list of strings (ie tab-split the actual VCF line),
# return [chromAsInt, Pos, End, SVType] of types [int, int, int, string]
# to allow easy comparisons
def line2sortable(line):
    chrom = line[0]
    # strip chr prefix if present
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    if chrom.isdigit():
        chrom = int(chrom)
    elif chrom == "X":
        chrom = 1023
    elif chrom == "Y":
        chrom = 1024
    elif (chrom == "M") or (chrom == "MT"):
        chrom = 1025
    else:
        # for other non-numeric chroms, use arbitrary numbers between 1100 and 2100
        chrom = 1100 + int(hashlib.sha256(chrom).hexdigest(), 16) % 1000

    pos = int(line[1])

    end = int(re.search(r"END=(\d+)$", line[7]).group(1))

    svType = line[4]

    return([chrom, pos, end, svType])
