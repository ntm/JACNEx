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
import logging
import math
import numpy
import os
import sys
import time

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

##########################################
# recalibrateGQs:
# recalibrate the GQs of a cluster with FITWITHs, in order to obtain similar
# numbers of calls of each CNVType as in its reference cluster.
#
# Args:
# - CNVs: list of CNVs, a CNV is a list (types [int, int, int, float, int]):
#   [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleIndex]
# - numSamples: number of samples in the cluster (not its ref cluster)
# - refVcfFile (str): name (with path) of the VCF file for the reference cluster (ie
#   cluster without FITWITHs) that serves as FITWITH for the current cluster, if any
# - minGQ: float, minimum Genotype Quality (GQ)
# - clusterID (str): id of current cluster (for logging)
#
# Returns the recalibrated CNVs.
def recalibrateGQs(CNVs, numSamples, refVcfFile, minGQ, clusterID):
    if refVcfFile == "":
        return(CNVs)
    recalCNVs = []
    maxCalls = countCallsFromVCF(refVcfFile)
    # GQs[CN] is the list of GQs of calls of type CN
    GQs = [[], [], [], []]
    for cnv in CNVs:
        GQs[cnv[0]].append(cnv[3])

    # minGQperCN[CN] is the min GQ that results in accepting at most maxCalls[CN] calls per
    # sample on average
    minGQperCN = [0, 0, 0, 0]
    for cn in (0, 1, 3):
        numAcceptedCalls = math.floor(numSamples * maxCalls[cn])
        # we want numAcceptedCalls to have GQs >= minGQ (so they are actually accepted)
        if numAcceptedCalls < len(GQs[cn]):
            sortedGQs = sorted(GQs[cn], reverse=True)
            minGQperCN[cn] = sortedGQs[numAcceptedCalls] - minGQ

    for cnv in CNVs:
        thisRecalGQ = cnv[3] - minGQperCN[cnv[0]]
        if thisRecalGQ > minGQ:
            # > rather than >= , because some non-ref clusters are actually too far from
            # their ref cluster on many exons and result in poor calls even at max GQ...
            # these get recalibrated by -(maxGQ-minGQ) (eg -98 with defaults maxGQ==100
            # and minGQ==2), but we still produce many (bogus) calls if testing with >=
            recalCNVs.append([cnv[0], cnv[1], cnv[2], thisRecalGQ, cnv[4]])

    if minGQperCN != [0, 0, 0, 0]:
        logger.info("cluster %s - recalibrated GQs by -%.1f (CN0), -%.1f (CN1), -%.1f (CN3+)",
                    clusterID, minGQperCN[0], minGQperCN[1], minGQperCN[3])

    return(recalCNVs)


##########################################
# printCallsFile:
# print CNVs for a single cluster in VCF format.
#
# Args:
# - outFile (str): name of VCF file to create. Will be squashed if pre-exists,
#   can include a path (which must exist), will be gzipped if outFile ends with '.gz'.
# - CNVs: list of CNVs, a CNV is a list (types [int, int, int, float, int]):
#   [CNVType, firstExonIndex, lastExonIndex, qualityScore, sampleIndex]
#   where firstExonIndex and lastExonIndex are indexes in the provided exons,
#   and sampleIndex is the index in the provided samples
# - FPMs: 2D-array of floats, size = nbExons * nbSamples, FPMs[e,s] is the FPM
#   count for exon e in sample s
# - CN2mus: 1D-array of nbExons floats, CN2mus[e] is the fitted mu of
#   the CN2 model of exon e for the cluster
# - Ecodes: 1D-array of nbExons bytes, exon is NOCALL if < 0
# - samples: list of nbSamples sampleIDs (==strings)
# - exons: list of nbExons exons, one exon is a list [CHR, START, END, EXONID]
# - BPDir: dir containing the breakpoint files
# - padding (int): padding bases used
# - madeBy (str): Name + version of program that made the CNV calls
# - minGQ: float, minimum Genotype Quality (GQ)
# - cnvPlotDir: string, subdir where plotFiles for each CNV were created (or "")
def printCallsFile(outFile, CNVs, FPMs, CN2Mus, Ecodes, samples, exons, BPDir, padding,
                   madeBy, minGQ, cnvPlotDir):
    # max GQ to produce in the VCF
    maxGQ = 100
    # min number of aligned fragments supporting given breakpoints to consider
    # them well-supported, hard-coded here, in our hands 3 or even 2 is strong
    # support but rarely seen, 1 may be more noisy but still useful
    minSupportingFrags = 1
    BPs = parseBreakpoints(BPDir, samples, minSupportingFrags)

    try:
        if outFile.endswith(".gz"):
            outFH = gzip.open(outFile, "xt", compresslevel=6)
        else:
            outFH = open(outFile, "x")
    except Exception as e:
        logger.error("Cannot (gzip-)open CNCallsFile %s: %s", outFile, e)
        raise Exception('cannot (gzip-)open CNCallsFile')

    # Header
    toPrint = "##fileformat=VCFv4.3\n"
    toPrint += "##fileDate=" + time.strftime("%y%m%d") + "\n"
    toPrint += "##source=" + madeBy + "\n"
    toPrint += "##JACNEx_commandLine=" + os.path.basename(sys.argv[0]) + ' ' + ' '.join(sys.argv[1:]) + "\n"
    # minGQ and cnvPlotDir lines must stay in sync with parsing lines in checkPrevVCFs(),
    # if these lines change remember to update checkPrevVCFs()
    toPrint += "##JACNEx_minGQ=" + str(minGQ) + "\n"
    toPrint += "##JACNEx_cnvPlotDir=" + cnvPlotDir + "\n"

    toPrint += """##ALT=<ID=DEL,Description="Deletion">
##ALT=<ID=DUP,Description="Duplication">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality, ie log10(P[sample has CNV] / P[sample is HOMOREF]) rounded to nearest int">
##FORMAT=<ID=FR,Number=1,Type=Float,Description="Fragment count Ratio, ie mean (over all overlapped called exons) of the ratio [FPM in sample] / [mean FPM of all HOMOREF samples in cluster]">
##FORMAT=<ID=BPR,Number=2,Type=String,Description="BreakPoint Ranges, ie ranges of coordinates containing the upstream and downstream breakpoints">
##FORMAT=<ID=BP,Number=.,Type=String,Description="Putative BreakPoints based on split reads (when available), formatted as BP1-BP2-N where N is the number of supporting fragments">
"""
    # COULD/SHOULD ADD:
    """
    ##reference=file:///path/to/hs38DH.fa
    ##contig=<ID=chr1,length=248956422>
    ##contig=...
    """

    colNames = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
    toPrint += "\t".join(colNames) + "\n"
    outFH.write(toPrint)

    # CNVs: sort by chrom-start-end (ie by exonIndexes), and at identical coords we will
    # create (up to) 2 VCF lines: for DELs then DUPs -> sort by CNVType
    CNVs.sort(key=lambda CNV: (CNV[1], CNV[2], CNV[0], CNV[4]))

    prevVcfStart = ""
    vcfGenos = []
    for CNV in CNVs:
        (cn, startExi, endExi, qualScore, sampleIndex) = CNV
        chrom = exons[startExi][0]

        # VCF spec says we must use POS = last base before the CNV
        pos = exons[startExi][1] - 1
        if (pos <= 0):
            pos = 0
            logger.warning("exon %s START <= padding, cannot remove padding -> using POS=0 in VCF",
                           exons[startExi][3])
        else:
            pos += padding

        end = exons[endExi][2] - padding

        # type of CNV
        if cn <= 1:
            svType = "DEL"
        else:
            svType = "DUP"
        alt = '<' + svType + '>'

        # VCF spec says we should use the ref genome's base, use N so we are compliant
        vcfStart = f"{chrom}\t{pos}\t.\tN\t{alt}\t.\tPASS\tSVTYPE={svType};END={end}\tGT:GQ:FR:BPR"

        if vcfStart != prevVcfStart:
            if prevVcfStart != "":
                toPrint = prevVcfStart + "\t" + "\t".join(vcfGenos) + "\n"
                outFH.write(toPrint)
            prevVcfStart = vcfStart
            # default to all-HomoRef
            vcfGenos = ["0/0"] * len(samples)

        # FragRatio: average of FPM ratios over all called exons in the CNV, calculate this
        # first because we need it for GT of DUPs
        fragRat = 0
        numExons = 0
        for ei in range(startExi, endExi + 1):
            if Ecodes[ei] >= 0:
                fragRat += FPMs[ei, sampleIndex] / CN2Mus[ei]
                numExons += 1
            # else exon ei is NOCALL, ignore it
        fragRat /= numExons

        # genotype, depending on fragRat for DUPs: at a diploid locus we
        # expect fragRat ~1.5 for HET-DUPs and starting at ~2 for CN=4+ (which we
        # will call HV-DUPs, although maybe CN=3 on one allele and CN=1 on the other)...
        # [NOTE: at a haploid locus we expect ~2 for a hemizygous DUP but we're OK
        # to call those HV-DUP]
        # We should therefore use an arbitrary cutoff between 1.5 and 2. However we
        # see a lot of "leakage" towards high FPMs, rsulting in too many
        # HV DUP calls that don't seem bona fide. So we use a high minFragRatDupHomo
        # to be conservative
        minFragRatDupHomo = 3.0
        geno = "1/1"
        if (cn == 1):
            geno = "0/1"
        elif (cn == 3) and (fragRat < minFragRatDupHomo):
            geno = "0/1"

        # cap GQ (will round to nearest int later)
        if qualScore > maxGQ:
            qualScore = maxGQ

        # BPR == s1-e1,s2-e2:
        e1 = pos
        s2 = end + 1
        # for s1 we need prev called exon
        prevCalled = startExi - 1
        while ((prevCalled > 0) and (Ecodes[prevCalled] < 0) and (exons[prevCalled][0] == chrom)):
            prevCalled -= 1
        if (exons[prevCalled][0] == chrom) and (Ecodes[prevCalled] >= 0):
            s1 = exons[prevCalled][2] + 1 - padding
        else:
            # no prev called exon on chrom, range starts at POS=0
            s1 = 0
        # similarly, for e2 we need the next called exon
        nextCalled = endExi + 1
        while ((nextCalled < len(Ecodes)) and (Ecodes[nextCalled] < 0) and (exons[nextCalled][0] == chrom)):
            nextCalled += 1
        if (nextCalled < len(Ecodes)) and (exons[nextCalled][0] == chrom):
            e2 = exons[nextCalled][1] - 1 + padding
        else:
            # no next called exon on chrom, range ends at end of last exon on chrom
            e2 = exons[nextCalled - 1][2] + 1 - padding
        bpRange = str(s1) + '-' + str(e1) + ',' + str(s2) + '-' + str(e2)

        # BP: find any split-read-supported breakpoints compatible with this CNV
        breakPoints = findBreakpoints(BPs, samples[sampleIndex], chrom, svType, s1, e1, s2, e2)

        # round GQ to nearest int and fragRat to 2 decimals
        vcfGenos[sampleIndex] = f"{geno}:{qualScore:.0f}:{fragRat:.2f}:{bpRange}"
        if (breakPoints != ""):
            # we have breakpoints for at least one sample
            if not prevVcfStart.endswith(':BP'):
                prevVcfStart += ':BP'
            vcfGenos[sampleIndex] += f":{breakPoints}"

    # print last CNV
    if prevVcfStart != "":
        toPrint = prevVcfStart + "\t" + "\t".join(vcfGenos) + "\n"
        outFH.write(toPrint)
    outFH.close()


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

##########################################
# countCallsFromVCF:
# count the number of calls per sample of each CN type, in vcfFile.
# Based on this, calculate a max expected number of calls per sample
# of each CN type for "similar" samples: the vcfFile holds calls made for
# a "reference" cluster, and the stats will be used to recalibrate the GQ
# scores of an associated cluster that uses this ref cluster as FITWITH,
# in order to obtain (at most) "similar" numbers of calls of each type.
# "Similar" here is hard-coded as the 80%-quantile of numbers of calls (of
# each type) from the ref cluster, see numCallsQuantile.
#
# Args:
# - vcfFile: string, name of an existing VCF file (with path)
#
# Returns: maxCalls: list of 4 ints, maxCalls[CN] is the max number of allowed CN calls
# per sample (maxCalls[2] is ignored).
def countCallsFromVCF(vcfFile):
    try:
        if vcfFile.endswith(".gz"):
            vcfFH = gzip.open(vcfFile, "rt")
        else:
            vcfFH = open(vcfFile, "r")
    except Exception as e:
        logger.error("Cannot open VCF file - %s", e)
        raise Exception("countCallsFromVCF failed, check log")

    for line in vcfFH:
        if line.startswith('#CHROM'):
            numSamples = len(line.rstrip().split("\t")) - 9
            CN0perSample = numpy.zeros(numSamples, dtype=numpy.uint32)
            CN1perSample = numpy.zeros(numSamples, dtype=numpy.uint32)
            CN3perSample = numpy.zeros(numSamples, dtype=numpy.uint32)
        elif not line.startswith('#'):
            calls = line.rstrip().split("\t")
            alt = calls[4]
            del calls[:9]
            for si in range(len(calls)):
                if calls[si].startswith('0/1:'):
                    if alt == '<DEL>':
                        CN1perSample[si] += 1
                    else:
                        CN3perSample[si] += 1
                elif calls[si].startswith('1/1'):
                    if alt == '<DEL>':
                        CN0perSample[si] += 1
                    else:
                        CN3perSample[si] += 1
    vcfFH.close()

    # accept up to hard-coded 80%-quantile of numbers of calls from the ref cluster
    numCallsQuantile = 0.8
    maxCalls = [0, 0, 0, 0]
    maxCalls[0] = numpy.quantile(CN0perSample, numCallsQuantile)
    maxCalls[1] = numpy.quantile(CN1perSample, numCallsQuantile)
    maxCalls[3] = numpy.quantile(CN3perSample, numCallsQuantile)
    return(maxCalls)


##########################################
# parseBreakpoints:
# parse the breakPoints files in subdir BPDir for all specified samples,
# filter out those supported by less than minSupportingFrags fragments==QNAMES,
# and return the others as BPs:
# dict, key==sample and value is a dict with key==chrom and value is a dict
# with key==svType and value is a list of lists [start, end, countQnames], ie:
# BPs[sample][chrom][svType] is a list of lists [start, end, countQnames].
def parseBreakpoints(BPDir, samples, minSupportingFrags):
    BPs = {}

    for sample in samples:
        # NOTE: keep filenames in sync with bpFile in s1_countFrags.py
        bpFile = os.path.join(BPDir, sample + '.breakPoints.tsv.gz')
        if (not os.path.isfile(bpFile)):
            logger.warning("cannot find breakPoints file %s for sample %s, this is unexpected... investigate?",
                           bpFile, sample)
            continue

        BPs[sample] = {}
        bpFH = gzip.open(bpFile, "rt")
        # check header
        if (bpFH.readline().rstrip() != "CHR\tSTART\tEND\tCNVTYPE\tCOUNT-QNAMES\tQNAMES"):
            logger.error("breakPoints file %s has unexpected header, FIX CODE", bpFile)
            raise Exception('bpFile bad header')

        for line in bpFH:
            try:
                (thisChrom, start, end, thisType, countQnames, qnames) = line.rstrip().split("\t")
                start = int(start)
                end = int(end)
                countQnames = int(countQnames)
            except Exception:
                logger.error("breakPoints file has line with wrong number of fields: %s", line)
                raise Exception('bpFile bad line')

            if (countQnames < minSupportingFrags):
                continue

            if thisChrom not in BPs[sample]:
                BPs[sample][thisChrom] = {}
            if thisType not in BPs[sample][thisChrom]:
                BPs[sample][thisChrom][thisType] = []
            BPs[sample][thisChrom][thisType].append([start, end, countQnames])
        bpFH.close()
    return(BPs)


##########################################
# findBreakpoints:
# look for split-read-supported breakpoint info that could support a CNV of
# type svtype, whose breakpoints lie in ranges s1-e1 and s2-e2.
# Breakpoints in BPs are as returned by parseBreakpoints().
# Returns all putative breakpoints as a single string:
# comma-separated BP1-BP2-N blocks, where N is the number of supporting fragments
def findBreakpoints(BPs, sample, chrom, svType, s1, e1, s2, e2):
    breakpoints = []
    if (sample in BPs) and (chrom in BPs[sample]) and (svType in BPs[sample][chrom]):
        for bpInfo in BPs[sample][chrom][svType]:
            (start, end, countQnames) = bpInfo
            if ((start >= s1) and (start <= e1) and (end >= s2) and (end <= e2)):
                breakpoints.append(f"{start}-{end}-{countQnames}")
            elif (start > e1):
                # breakpoints in BPFiles are sorted by chrom then start then...
                break
    return(','.join(breakpoints))
