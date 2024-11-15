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


import concurrent.futures
import logging
import numba  # make python faster
import numpy
import os
import re
import subprocess
import tempfile

####### JACNEx modules
import countFrags.bed


# silence numba when we are in DEBUG mode
logging.getLogger('numba').setLevel(logging.INFO)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


# define global dictionary of NCLs.
# This must be a (module-level) global variable because the multiprocessing
# module doesn't allow us to use an NCL as function argument...
# So, users of this module MUST call initExonNCLs() once to populate exonNCLs
# before the first call to bam2counts()
exonNCLs = {}


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

####################################################
# initExonNCLs:
# Create nested containment lists (similar to interval trees but faster), one per
# chromosome, representing the exons
# Arg: exon definitions as returned by processBed, padded and sorted
# Resulting NCLs are stored in the global dictionary exonNCLs, one NCL per
# chromosome, key=chr, value=NCL
# This function must be called a single time before the first call to bam2counts()
def initExonNCLs(exons):
    # we want to access the module-global exonNCLs dictionary
    global exonNCLs
    if exonNCLs:
        logger.error("initExonNCLs called but exonNCLs already initialized, fix your code")
        raise Exception("initExonNCLs() called twice")
    exonNCLs = countFrags.bed.buildExonNCLs(exons)


####################################################
# bam2counts :
# Count the fragments in bamFile that overlap each exon from exonNCLs.
# Arguments:
#   - a bam file (with path)
#   - total number of exons
#   - the maximum accepted gap length between paired reads
#   - a fast tmp dir with enough space for samtools collate
#   - the samtools binary, with path
#   - number of cpu cores that we can use
#   - an int (sampleIndex) that is not used but is simply returned (for multiprocessing)
#
# Pre-requirement: initExonNCLs must have been called to populate exonNCLs before
# the first call to this function.
#
# Return a 3-element tuple (sampleIndex, sampleCounts, breakPointsTSV) where:
# - sampleCounts is a 1D numpy int array dim = nbOfExons allocated here and filled with
#   the counts for this sample,
# - breakPointsTSV is a single string storing data in TSV format with:
#   CHR START END CNVTYPE COUNT-QNAMES QNAMES
#   where CHR-START-END are the coordinates of the putative CNV, CNVTYPE is 'DEL' or 'DUP',
#   COUNT-QNAMES is the number of QNAMEs that support this CNV, and QNAMES is the
#   comma-separated list of supporting QNAMEs.
# If anything goes wrong, log info on exception and then always raise Exception(str(sampleIndex)),
# so caller can catch it and know which sampleIndex we were working on.
def bam2counts(bamFile, nbOfExons, maxGap, tmpDir, samtools, jobs, sampleIndex):
    # This is a two step process:
    # 1. group the alignments by QNAME with samtools-collate, and then
    # 2. split the alignments into batches of ~batchSize alignements (making sure all alis for any
    #    QNAME are in the same batch), and process each batch in parallel (with processBatch())
    # Since samtools-collate needs to be mostly done before step 2 starts, we just use
    # max(jobs-1,1) as both the number of samtools threads (step 1) and the number of
    # python processes (step 2). This means we may use slightly more than job cores
    # simultaneously, tune it down if needed.
    realJobs = max(jobs - 1, 1)
    # number of alignments to process in a batch, this is just a performance tuning param,
    # default should be OK.
    batchSize = 1000000
    # min MAPQ to accept alignments
    minMapq = 20

    try:
        # data structures to return:
        # 1D numpy array containing the sample fragment counts for all exons
        sampleCounts = numpy.zeros(nbOfExons, dtype=numpy.uint32)
        # list of lists with info about breakpoints support
        breakPoints = []

        ############################################
        # Preprocessing:
        # - need to parse the alignements grouped by qname, "samtools collate" allows this;
        # - we can immediately filter out poorly mapped (low MAPQ) or dubious/bad
        #   alignments based on the SAM flags
        # - we also ignore alignments on contigs that don't carry exons (eg decoy, but also
        #   GRCh38 ALT contigs: samples in the same cluster can match different ALT haplotypes,
        #   therefore counting alis on ALTs would result in bad CNV calls), and we "ignore"
        #   the MAPQ changes done by bwa-postalt if it ran and reduced the MAPQ on the main
        #   chroms in favor of ALTs (we use the "om" tag, which contains the original mapq)
        # Requiring:
        #   1 0x1 read paired
        # Discarding when any if the below is set:
        #   4 0x4 read unmapped
        #   8 0x8 mate unmapped
        #   256 0x100 secondary alignment
        #   512 0x200 read fails platform/vendor quality checks
        #   1024 0x400 read is PCR or optical duplicate
        #   -> sum == 1804
        # For more details on FLAGs read the SAM spec: http://samtools.github.io/hts-specs/
        tmpDirObj = tempfile.TemporaryDirectory(dir=tmpDir)
        tmpDirPrefix = tmpDirObj.name + "/tmpcoll"

        cmd = [samtools, 'collate', '-O', '--output-fmt', 'SAM', '--threads', str(realJobs)]
        filterString = 'filter=flag.paired && !(flag & 1804) && ((mapq >= ' + minMapq
        filterString += ') || ([om] >= ' + minMapq + ')) && (rname =~ "^('
        filterString += '|'.join(exonNCLs.keys()) + ')$")'
        cmd.extend(['--input-fmt-option', filterString])
        cmd.extend([bamFile, tmpDirPrefix])
        samproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

        #####################################################
        # Define nested callback for processing processBatch() result (so sampleCounts
        # and breakPoints are in its scope)

        # batchDone:
        # arg: a Future object returned by ProcessPoolExecutor.submit(processBatch).
        # processBatch() returns a 2-element tuple (batchCounts, batchBPs).
        # If something went wrong, log and propagate exception;
        # otherwise add batchCounts to sampleCounts (these are np arrays of same size),
        # and append batchBPs to breakPoints
        def batchDone(futureProcBatchRes):
            e = futureProcBatchRes.exception()
            if e is not None:
                #  not expecting any exceptions, if any it should be fatal for bamFile
                logger.error("Failed to process a batch for %s", os.path.basename(bamFile))
                raise(e)
            else:
                procBatchRes = futureProcBatchRes.result()
                # nonlocal declaration means += updates sampleCounts from enclosing scope
                nonlocal sampleCounts
                sampleCounts += procBatchRes[0]
                breakPoints.extend(procBatchRes[1])

        ############################################
        # parse alignments
        with concurrent.futures.ProcessPoolExecutor(realJobs) as pool:
            # next line to examine
            nextLine = samproc.stdout.readline()

            # skip header
            while re.match('^@', nextLine):
                nextLine = samproc.stdout.readline()

            # store lines until batchSize alis were parsed, then process batch
            thisBatch = []
            thisBatchSize = 0
            prevQname = ''

            while nextLine != '':
                if (thisBatchSize < batchSize):
                    thisBatch.append(nextLine)
                    thisBatchSize += 1
                else:
                    # grab QNAME of nextLine (with trailing \t, it doesn't matter)
                    thisQname = re.match(r'[^\t]+\t', nextLine).group()
                    if (prevQname == ''):
                        # first ali after filling this batch
                        thisBatch.append(nextLine)
                        prevQname = thisQname
                    elif (thisQname == prevQname):
                        # batch is full but same QNAME
                        thisBatch.append(nextLine)
                    else:
                        # this batch is full and we changed QNAME => ''-terminate batch (as expected by
                        # processBatch), process it and start a new batch
                        thisBatch.append('')
                        futureRes = pool.submit(processBatch, thisBatch, nbOfExons, maxGap, minMapq)
                        futureRes.add_done_callback(batchDone)
                        thisBatch = [nextLine]
                        thisBatchSize = 1
                        prevQname = ''
                # in all cases, read next line
                nextLine = samproc.stdout.readline()
            # process last batch (unless bamFile had zero alignments)
            if thisBatchSize > 0:
                thisBatch.append('')
                futureRes = pool.submit(processBatch, thisBatch, nbOfExons, maxGap, minMapq)
                futureRes.add_done_callback(batchDone)

        # wait for samtools to finish cleanly and check exit code
        if (samproc.wait() != 0):
            logger.error("in bam2counts, while processing %s, samtools exited with code %s",
                         bamFile, samproc.returncode)
            raise Exception("samtools failed")

        # tmpDirObj should get cleaned up automatically but sometimes samtools tempfiles
        # are in the process of being deleted => sync to avoid race
        os.sync()
        # sort by chrom then start then end then...
        countFrags.bed.sortExonsOrBPs(breakPoints)
        # count QNAMES supporting the same breakpoints and produce BP data as TSV
        breakPointsTSV = countAndMergeBPs(breakPoints)
        return(sampleIndex, sampleCounts, breakPointsTSV)

    except Exception as e:
        logger.error("bam2counts failed for %s - %s", bamFile, repr(e))
        raise Exception(str(sampleIndex))


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

####################################################
# aliLengthOnRef:
# Arg: a CIGAR string
# Returns the span of the alignment on the reference sequence, ie number of bases
# consumed by the alignment on the reference
def aliLengthOnRef(cigar):
    length = 0
    # only count CIGAR operations that consume the reference sequence, see CIGAR definition
    # in SAM spec available here: https://samtools.github.io/hts-specs/
    match = re.findall(r"(\d+)[MDN=X]", cigar)
    for op in match:
        length += int(op)
    return(length)


####################################################
# firstNonClipped:
# Arg: a CIGAR string
# Returns the coordinate on the read of the leftmost non-clipped base
def firstNonClipped(cigar):
    firstNonClipped = 1
    # count all leading H and S bases
    while True:
        match = re.match(r"^(\d+)[HS]", cigar)
        if match:
            firstNonClipped += int(match.group(1))
            cigar = re.sub(r"^(\d+)[HS]", '', cigar)
        else:
            break
    return(firstNonClipped)


####################################################
# Arg: sorted list of "breakpoints" == [CHR, START, END, CNVTYPE, QNAME]
#
# Count the number of consecutive lines with the same CHR-START-END-CNVTYPE,
# and merge these into a single [CHR, START, END, CNVTYPE, COUNT-QNAMES, QNAMES].
# Return all the resulting data as a single string storing a TSV (including
# header, \t field separators and \n line terminations).
def countAndMergeBPs(breakPoints):
    allBPs = "CHR\tSTART\tEND\tCNVTYPE\tCOUNT-QNAMES\tQNAMES\n"
    # append dummy BP to end the loop properly
    breakPoints.append(['DUMMY', 0, 0, 'DUMMY', 'DUMMY'])
    # accumulators, rely on the fact that the breakPoints are sorted
    start = ""
    count = 0
    qnames = []
    for bp in breakPoints:
        thisStart = bp[0] + "\t" + str(bp[1]) + "\t" + str(bp[2]) + "\t" + bp[3]
        if (thisStart == start):
            count += 1
            qnames.append(bp[4])
        else:
            if count != 0:
                allBPs += start + "\t" + str(count) + "\t" + ",".join(qnames) + "\n"
            start = thisStart
            count = 1
            qnames = [bp[4]]
    return(allBPs)


####################################################
# processBatch :
# Count the fragments in batchOfLines that overlap each exon from exonNCLs.
# Arguments:
#   - a batch=list of SAM lines grouped by QNAME. All alis for any given QNAME
#     must be in the same batch. A batch must end with an empty line (ie '').
#   - total number of exons
#   - the maximum accepted gap length between paired reads
#   - the min MAPQ/om applied to keep an alignment (for ALT-aware processing)
# Return a 2-element tuple (batchCounts, batchBPs) where:
# - batchCounts is a 1D numpy int array dim = nbOfExons allocated here and filled with
#   the counts for this batch of lines,
# - batchBPs is a list of 5-element lists [CHR, START, END, CNVTYPE, QNAME], where
#   START and END are the coordinates of the putative breakpoints supported by this
#   batch of lines, CNVTYPE is 'DEL' or 'DUP',  and QNAME is the supporting fragment
def processBatch(batchOfLines, nbOfExons, maxGap, minMapq):
    # We need to process all alis for a QNAME together
    # -> store data in accumulators until we change QNAME
    # Accumulators:
    # QNAME and CHR
    qname, qchrom = "", ""
    # START and END coordinates on the genome of each alignment for this qname,
    # aligning on the Forward or Reverse genomic strands
    qstartF, qstartR, qendF, qendR = [], [], [], []
    # coordinate of the leftmost non-clipped base on the read, alis in the same order
    # as in qstartF/R and qendF/R
    qstartOnReadF, qstartOnReadR = [], []
    # qFirstOnForward==1 if the first-in-pair read of this qname is on the
    # forward reference strand, -1 if it's on the reverse strand, 0 if we don't yet know
    qFirstOnForward = 0
    # qBad==True if qname must be skipped (e.g. alis on multiple chroms, or alis
    # disagree regarding the strand on which the first/last read-in-pair aligns, or...)
    qBad = False

    # results to return:
    # 1D numpy array containing the fragment counts for all exons in batchOfLines
    batchCounts = numpy.zeros(nbOfExons, dtype=numpy.uint32)
    # list of lists with info about breakpoints support
    batchBPs = []

    for line in batchOfLines:
        ali = []
        if (line != ''):
            ali = line.split('\t')
        if (len(ali) == 0) or (ali[0] != qname):
            # we are done with batchOfLines or we changed qname: process accumulated data
            if (not qBad) and (qname != "") and (qchrom != ""):
                # (qchrom == "" is possible if all alis for qname had low MAPQ and were rescued
                # by acceptable om, but actually mapped as SA/XA to non-ALT contigs)
                # if we have 2 alis on a strand, make sure they are in "read" order (switch them if needed)
                if (len(qstartF) == 2) and (qstartOnReadF[0] > qstartOnReadF[1]):
                    qstartF.reverse()
                    qendF.reverse()
                if (len(qstartR) == 2) and (qstartOnReadR[0] > qstartOnReadR[1]):
                    qstartR.reverse()
                    qendR.reverse()
                BPs = Qname2ExonCount(qstartF, qendF, qstartR, qendR, qchrom, batchCounts, maxGap)
                if (BPs is not None):
                    BPs.insert(0, qchrom)
                    BPs.append(qname)
                    batchBPs.append(BPs)
            if (len(ali) == 0):
                # ali == empty list means we are done with this batch
                return(batchCounts, batchBPs)
            else:
                # reset accumulators
                qname, qchrom = "", ""
                qstartF, qstartR, qendF, qendR = [], [], [], []
                qstartOnReadF, qstartOnReadR = [], []
                qFirstOnForward = 0
                qBad = False
        elif qBad:  # same qname as previous ali but we know it's bad -> skip
            continue

        ######################################################################
        # Either we're in the same qname and it's not bad, or we changed qname
        # -> in both cases update accumulators with current line
        if qname == "":
            qname = ali[0]
            qchrom = ali[2]
        elif qchrom != ali[2]:
            # qname has alignments on different chroms, ignore it
            qBad = True
            continue

        if ali[4] < minMapq:
            # MAPQ low but om acceptable: we only keep the alignment if all SA and XA
            # alignements are on ALT contigs
            # optional fiends in SAM start at column 12, ie ali[11]
            for optF in ali[11:]:
                match = re.match(r"^[SX]A:Z:(.+)$", optF)
                if match:
                    otherAlis = match.group(1).split(';')
                    # XAs/SAs always end with ';' -> discard last (empty) element
                    otherAlis.pop()
                    for oa in otherAlis:
                        if re.match(r"^[^,]+_alt,", oa) is None:
                            qBad = True
                            break
                    if qBad:
                        # don't examine other optF's
                        break
            if qBad:
                # ignore this ali
                continue

        # Retrieving flags for STRAND and first/second read
        currentFlag = int(ali[1])

        # currentFirstOnForward==1 if according to this ali, the first-in-pair
        # read is on the forward strand, -1 otherwise
        currentFirstOnForward = -1
        # flag 16 the alignment is on the reverse strand
        # flag 64 the alignment is first in pair, otherwise 128
        if ((currentFlag & 80) == 64) or ((currentFlag & 144) == 144):
            currentFirstOnForward = 1
        if qFirstOnForward == 0:
            # first ali for this qname
            qFirstOnForward = currentFirstOnForward
        elif qFirstOnForward != currentFirstOnForward:
            qBad = True
            continue
        # else this ali agrees with previous alis for qname -> NOOP

        # START and END coordinates of current alignment on REF (ignoring any clipped bases)
        currentStart = int(ali[3])
        # END depends on CIGAR
        currentCigar = ali[5]
        currentAliLength = aliLengthOnRef(currentCigar)
        currentEnd = currentStart + currentAliLength - 1
        # coord of leftmost non-clipped base on read
        currentStartOnRead = firstNonClipped(currentCigar)
        if (currentFlag & 16) == 0:
            # flag 16 off => forward strand
            qstartF.append(currentStart)
            qendF.append(currentEnd)
            qstartOnReadF.append(currentStartOnRead)
        else:
            qstartR.append(currentStart)
            qendR.append(currentEnd)
            qstartOnReadR.append(currentStartOnRead)

    ###########
    # Done reading lines from batchOfLines, and we already returned if the batch
    # ended with '' as it should, check this
    raise Exception("processBatch didn't return, batch wasn't ''-terminated?")


####################################################
# Qname2ExonCount :
# Given data representing all alignments for a single qname (must all map
# to the same chrom):
# - apply QC filters to ignore qnames that are impossible to interpret;
# - identify the genomic intervals that are covered by the sequenced fragment;
# - identify exons overlapped by these intervals, and increment their count.
# Args:
#   - 4 lists of ints for F and R strand positions (start and end), in "read" order
#   - the chromosome where current alignments map
#   - the counts vector to update (1D numpy int array)
#   - the maximum accepted gap length between a pair of mate reads, pairs separated by
#     a longer gap are ignored (putative structural variant or alignment error)
# This function updates countsVec, and returns a 3-element list [START, END, CNVTYPE] if
# the QNAME supports the presence of a CNV (CNVTYPE == 'DEL' or 'DUP') with breakpoints
# START & END (ie a read spans the breakpoints), None otherwise.
def Qname2ExonCount(startFList, endFList, startRList, endRList, chrom, countsVec, maxGap):
    # skip Qnames that don't have alignments on both strands
    if (len(startFList) == 0) or (len(startRList) == 0):
        return
    # skip Qnames that have at least 3 alignments on a strand
    if (len(startFList) > 2) or (len(startRList) > 2):
        return

    # declare list that will be returned
    breakPoints = None

    # if we have 2 alis on one strand and they overlap: merge them
    # (there could be a short indel or other SV, not trying to detect
    # this but we still cover this genomic region with this qname)
    if (len(startFList) == 2) and (min(endFList) > max(startFList)):
        startFList = [min(startFList)]
        endFList = [max(endFList)]
    if (len(startRList) == 2) and (min(endRList) > max(startRList)):
        startRList = [min(startRList)]
        endRList = [max(endRList)]

    #######################################################
    # identify genomic regions covered by the fragment
    #######################################################
    # fragStarts, fragEnds: start and end coordinates of the genomic intervals
    # covered by this fragment.
    # There is usually one (1F1R overlapping) or 2 intervals (1F1R not
    # overlapping) , but if the fragment spans a DEL there can be up to 3
    fragStarts = []
    fragEnds = []

    ##########################
    # 1F1R
    if (len(startFList) == 1) and (len(startRList) == 1):
        if (startFList[0] < endRList[0]):
            # face to face
            if (endFList[0] >= startRList[0]):
                # overlap => merge into a single interval
                fragStarts = [min(startFList[0], startRList[0])]
                fragEnds = [max(endFList[0], endRList[0])]
            elif (startRList[0] - endFList[0] <= maxGap):
                # no overlap but small gap between reads, assume no-CNV with library
                # fragment slightly longer than readLength*2 -> 2 intervals
                fragStarts = [startFList[0], startRList[0]]
                fragEnds = [endFList[0], endRList[0]]
            else:
                # large gap between reads: could be a DEL but without spanning reads,
                # so no clear evidence -> ignore qname
                return
        else:
            # back to back, could be a DUP but no spanning reads -> ignore qname
            return

    ##########################
    # 2F1R
    elif (len(startFList) == 2) and (len(startRList) == 1):
        if (startFList[1] <= endRList[0]) and (endFList[1] >= startRList[0]):
            # second F ali overlaps with R ali, build an interval by merging them
            fragStarts = [min(startFList[1], startRList[0])]
            fragEnds = [max(endFList[1], endRList[0])]
        elif (startFList[1] <= endRList[0]) and (startRList[0] - endFList[1] <= maxGap):
            # second F ali and R ali are face-to-face, they don't overlap but gap
            # is small -> build 2 intervals
            fragStarts = [startFList[1], startRList[0]]
            fragEnds = [endFList[1], endRList[0]]
        else:
            # second F ali and R ali are face-to-face but far apart, or they are back-to-back,
            # in both cases we can't interpret -> ignore qname
            return

        if (startFList[0] < startFList[1]):
            # F alis are in read order: spanning a DEL?
            if (endFList[0] < fragStarts[0]):
                # indeed this looks like a DEL, with first F ali an interval on its own (on the left)
                fragStarts.insert(0, startFList[0])
                fragEnds.insert(0, endFList[0])
                # this is a suspected DEL with BPs: fragEnds[0], fragStarts[1]
                breakPoints = [fragEnds[0], fragStarts[1], 'DEL']
            else:
                # Both F alis are overlapped by R ali, doesn't make sense, ignore this qname
                return
        else:
            # F alis are inverted, ie beginning of read aligns after end of read on
            # the genome: spanning a DUP?
            if (fragEnds[-1] < startFList[0]):
                # indeed this looks like a DUP, with first F ali (==start of F read) an interval on
                # its own (on the right)
                fragStarts.append(startFList[0])
                fragEnds.append(endFList[0])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            elif (fragStarts[-1] < endFList[0]):
                # first F ali overlaps R ali, very small DUP and/or large fragment? suspicious, but merge them
                fragStarts[-1] = min(fragStarts[-1], startFList[0])
                fragEnds[-1] = max(fragEnds[-1], endFList[0])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            else:
                # F alis suggest a DUP but R ali doesn't agree, ignore this qname
                return

    ##########################
    # 1F2R
    elif (len(startFList) == 1) and (len(startRList) == 2):
        if (startFList[0] <= endRList[1]) and (endFList[0] >= startRList[1]):
            # second R ali (in read order) overlaps with F ali, build an interval by merging them
            fragStarts = [min(startFList[0], startRList[1])]
            fragEnds = [max(endFList[0], endRList[1])]
        elif (startFList[0] <= endRList[1]) and (startRList[1] - endFList[0] <= maxGap):
            # second R ali and F ali are face-to-face, they don't overlap but gap
            # is small -> build 2 intervals
            fragStarts = [startFList[0], startRList[1]]
            fragEnds = [endFList[0], endRList[1]]
        else:
            # second R ali and F ali are face-to-face but far apart, or they are back-to-back,
            # in both cases we can't interpret -> ignore qname
            return

        if (startRList[1] < startRList[0]):
            # R alis are in order: spanning a DEL?
            if (fragEnds[-1] < startRList[0]):
                # indeed this looks like a DEL, with first R ali an interval on its own (on the right)
                fragStarts.append(startRList[0])
                fragEnds.append(endRList[0])
                # this is a suspected DEL with BPs: fragEnds[-2], fragStarts[-1]
                breakPoints = [fragEnds[-2], fragStarts[-1], 'DEL']
            else:
                # Both R alis are overlapped by F ali, doesn't make sense, ignore this qname
                return
        else:
            # R alis are inverted: spanning a DUP?
            if (endRList[0] < fragStarts[0]):
                # indeed this looks like a DUP, with first R ali (==start of R read) an interval on
                # its own (on the left)
                fragStarts.insert(0, startRList[0])
                fragEnds.insert(0, endRList[0])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            elif (fragStarts[0] < endRList[0]):
                # first R ali overlaps F ali, very small DUP and/or large fragment? suspicious, but merge them
                fragStarts[0] = min(fragStarts[0], startRList[0])
                fragEnds[0] = max(fragEnds[0], endRList[0])
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            else:
                # R alis suggest a DUP but F ali doesn't agree, ignore this qname
                return

    ##########################
    # 2F2R
    elif(len(startFList) == 2) and (len(startRList) == 2):
        if (startFList[0] < startFList[1]) and (startRList[0] > startRList[1]):
            # both F and R pairs of alis are in order: spanning a DEL?
            if ((startFList[0] <= endRList[1]) and (endFList[0] >= startRList[1]) and
                (startFList[1] <= endRList[0]) and (endFList[1] >= startRList[0])):
                # OK: leftmost F+R reads overlap and same for rightmost F+R reads -> merge each pair
                fragStarts = [min(startFList[0], startRList[1]), min(startFList[1], startRList[0])]
                fragEnds = [max(endFList[0], endRList[1]), max(endFList[1], endRList[0])]
                # this is a suspected DEL with BPs: fragEnds[0], fragStarts[1]
                breakPoints = [fragEnds[0], fragStarts[1], 'DEL']
            else:
                # at least one pair of F+R reads doesn't overlap, how? ignore qname
                return

        elif (startFList[0] > startFList[1]) and (startRList[0] < startRList[1]):
            # both F and R pairs of alis are inverted: spanning a DUP?
            if ((startFList[1] <= endRList[0]) and (endFList[1] >= startRList[0]) and
                (startFList[0] <= endRList[1]) and (endFList[0] >= startRList[1])):
                # OK: leftmost F+R reads overlap and same for rightmost F+R reads -> merge each pair
                fragStarts = [min(startFList[1], startRList[0]), min(startFList[0], startRList[1])]
                fragEnds = [max(endFList[1], endRList[0]), max(endFList[0], endRList[1])]
                # this is a suspected DUP with BPs: fragStarts[0], fragEnds[-1]
                breakPoints = [fragStarts[0], fragEnds[-1], 'DUP']
            else:
                # at least one pair of F+R reads doesn't overlap, how? ignore qname
                return

        else:
            # one pair is in order but the other is not, doesn't make sense
            return

    #######################################################
    # find exons overlapped by the fragment and increment counters
    #######################################################
    # we want to increment countsVec at most once per exon, even if
    # we have several intervals that overlap the same exon
    exonsSeen = []
    for fi in range(len(fragStarts)):
        overlappedExons = exonNCLs[chrom].find_overlap(fragStarts[fi], fragEnds[fi])
        for exon in overlappedExons:
            exonIndex = exon[2]
            if (exonIndex not in exonsSeen):
                exonsSeen.append(exonIndex)
                incrementCount(countsVec, exonIndex)
    return(breakPoints)


####################################################
# incrementCount:
# increment the counter in countsVec (1D numpy array) at index exonIndex
@numba.njit
def incrementCount(countsVec, exonIndex):
    countsVec[exonIndex] += 1
