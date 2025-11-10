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


import logging
import numpy

####### JACNEx modules
import countFrags.bed

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

#############################################################
# extractCountsFromPrev:
# Args:
#   - genomicWindows: exon and pseudo-exon definitions as returned by processBed
#   - SOIs: list of samples of interest (ie list of strings)
#   - prevCountsFile: a countsFile (npz format) produced by printCountsFile for some
#     samples (hopefully some of the SOIs), using the same (pseudo-)exon definitions
#     as in 'genomicWindows', if such a file is available; or '' otherwise
#
# Returns a tuple (countsArray, countsFilled), each is created here:
#   - countsArray is an int numpy 2D-array, size = nbGenomicWindows x nbSOIs,
#   initially all-zeroes
#   - countsFilled is a 1D boolean numpy array, size = nbSOIs, initially all-False
#
# If prevCountsFile=='' return the (all-zeroes/all-False) arrays
# Otherwise:
# -> make sure prevCountsFile was produced with the same BED+padding as genomicWindows,
#    else raise exception;
# -> for any sample present in both prevCounts and SOIs, fill the sample's column in
#    countsArray by copying data from prevCounts, and set countsFilled[sample] to True
def extractCountsFromPrev(genomicWindows, SOIs, prevCountsFile):
    # numpy arrays to be returned:
    # countsArray[exonIndex,sampleIndex] will store the specified count
    # order=F should improve performance, since we fill the array one column
    # at a time when parsing BAMs
    # dtype=numpy.uint32 should be fast and sufficient
    countsArray = numpy.zeros((len(genomicWindows), len(SOIs)), dtype=numpy.uint32, order='F')
    # countsFilled: same size and order as sampleNames, value will be set
    # to True iff counts were filled from countsFile
    countsFilled = numpy.zeros(len(SOIs), dtype=bool)

    if (prevCountsFile != ''):
        # we have a prevCounts file, parse it
        (prevGenomicWindows, prevSamples, prevCounts) = parseCountsFile(prevCountsFile)
        # compare genomicWindows definitions
        if (genomicWindows != prevGenomicWindows):
            logger.error("(pseudo-)exon definitions disagree between prevCountsFile and BED, " +
                         "countsFiles cannot be re-used if the BED file or padding changed")
            raise Exception('mismatched genomicWindows')

        # prevIndexes: temp dict, key = sample identifier, value = index in prevSamples
        prevIndexes = {}
        for prevIndex in range(len(prevSamples)):
            prevIndexes[prevSamples[prevIndex]] = prevIndex
        for newIndex in range(len(SOIs)):
            newSample = SOIs[newIndex]
            if newSample in prevIndexes:
                countsArray[:, newIndex] = prevCounts[:, prevIndexes[newSample]]
                countsFilled[newIndex] = True

    return(countsArray, countsFilled)


#############################################################
# parseAndNormalizeCounts:
# Parse the counts data in countsFile, normalize (see NOTE) as fragments per
# million (FPM), and return the results separately for exons and intergenic
# pseudo-exons.
# NOTE: for autosome exons FPM normalization is performed taking into account
# ONLY these autosome exon counts; for exons on sex chromosomes it is performed
# taking into account autosome AND gonosome counts; and for intergenic pseudo-exons
# it is performed taking into account ALL exonic and intergenic counts.
# This strategy avoids skewing autosome FPMs in men vs women (due to more reads on
# chrX in women), preserves ~2x more FPMs in women vs men for chrX exons, and
# results in intergenic FPMs correctly representing the amount of "intergenic noise"
# in each sample.
#
# Arg:
#   - a countsFile produced by printCountsFile (npz format)
#
# Returns a tuple (samples, autosomeExons, gonosomeExons, intergenics,
#                  autosomeFPMs, gonosomeFPMs, intergenicFPMs),
# each is created here and populated from countsFile data:
# -> 'samples' is the list of sampleIDs (strings)
# -> 'autosomeExons', 'gonosomeExons' and 'intergenics' are lists of
#    autosome/gonosome/intergenic (pseudo-)exons as produced by processBed
#    (EXON_ID is used to decide whether each window is an exon or an intergenic pseudo-exon)
# -> 'autosomeFPMs', 'gonosomeFPMs' and 'intergenicFPMs' are numpy 2D-arrays of floats,
#    of sizes [len(autosomeExons) | len(gonosomeExons) | len(intergenics)] x len(samples),
#    holding the FPM-normalized counts for autosome | gonosome | intergenic (pseudo-)exons
def parseAndNormalizeCounts(countsFile):
    (genomicWindows, samples, counts) = parseCountsFile(countsFile)
    sexChroms = countFrags.bed.sexChromosomes()

    # First pass: identify autosome/gonosome/intergenic (pseudo-)exons and populate
    # *Exons and intergenics
    autosomeExons = []
    gonosomeExons = []
    intergenics = []
    # windowType==0 for autosome exons, 1 for gonosome exons, 2 for intergenic pseudo-exons
    windowType = numpy.zeros(len(genomicWindows), dtype=numpy.uint8)

    for i in range(len(genomicWindows)):
        if genomicWindows[i][3].startswith("intergenic_"):
            intergenics.append(genomicWindows[i])
            windowType[i] = 2
        elif genomicWindows[i][0] in sexChroms:
            gonosomeExons.append(genomicWindows[i])
            windowType[i] = 1
        else:
            autosomeExons.append(genomicWindows[i])
            windowType[i] = 0

    # Second pass: populate *FPMs
    autosomeFPMs = counts[windowType == 0, :].astype(numpy.float64, order='F', casting='safe')
    gonosomeFPMs = counts[windowType == 1, :].astype(numpy.float64, order='F', casting='safe')
    intergenicFPMs = counts[windowType == 2, :].astype(numpy.float64, order='F', casting='safe')

    sumOfCountsAuto = autosomeFPMs.sum(dtype=numpy.float64, axis=0)
    sumOfCountsAutoGono = gonosomeFPMs.sum(dtype=numpy.float64, axis=0)
    sumOfCountsAutoGono += sumOfCountsAuto
    sumOfCountsTotal = intergenicFPMs.sum(dtype=numpy.float64, axis=0)
    sumOfCountsTotal += sumOfCountsAutoGono
    # if any sample has sumOfCounts*==0, replace by 1 to avoid dividing by zero
    sumOfCountsAuto[sumOfCountsAuto == 0] = 1.0
    sumOfCountsAutoGono[sumOfCountsAutoGono == 0] = 1.0
    sumOfCountsTotal[sumOfCountsTotal == 0] = 1.0
    # scale to get FPMs
    sumOfCountsAuto /= 1e6
    sumOfCountsAutoGono /= 1e6
    sumOfCountsTotal /= 1e6

    autosomeFPMs /= sumOfCountsAuto
    gonosomeFPMs /= sumOfCountsAutoGono
    intergenicFPMs /= sumOfCountsTotal

    return(samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs, intergenicFPMs)


#############################################################
# countsFile2samples:
# useful to quickly extract just the samples from a countsFile.
#
# Arg:
#   - a countsFile produced by printCountsFile()
#
# Returns the list of sampleIDs (ie strings) present in countsFile
def countsFile2samples(countsFile):
    try:
        npzCounts = numpy.load(countsFile, mmap_mode='r')
    except Exception as e:
        logger.error("Grabbing samples from provided countsFile %s: %s", countsFile, e)
        raise Exception('cannot grab samples from countsFile')

    # samples
    samples = (npzCounts['samples']).tolist()
    npzCounts.close()
    return(samples)


#############################################################
# printCountsFile:
# Args:
# - 'genomicWindows' is a list of exons and pseudo-exons as returned by processBed,
# ie each (pseudo-)exon is a list of 4 scalars (types: str,int,int,str) containing
# CHR,START,END,EXON_ID
# - 'samples' is a list of sampleIDs
# - 'counts' is a numpy 2D-array of uint32 of size nbWindows * nbSamples
# - 'outFile' is a filename with path, path must exist and file will be squashed if
# it pre-exist
#
# Print this data to outFile as a 'countsFile' (as parsed by parseCountsFile)
def printCountsFile(genomicWindows, samples, counts, outFile):
    try:
        numpy.savez_compressed(outFile, exons=genomicWindows, samples=samples, counts=counts)
    except Exception as e:
        logger.error("Cannot save counts to outFile %s: %s", outFile, e)
        raise Exception('cannot save counts to outFile')


###############################################################################
############################ PRIVATE FUNCTIONS ################################
###############################################################################

#############################################################
# parseCountsFile:
# Arg:
#   - a countsFile produced by printCountsFile()
#
# Returns a tuple (genomicWindows, samples, counts), each is created here:
# -> 'genomicWindows' is a list of exons and pseudo-exons as returned by processBed,
#    ie each (pseudo-)exon is a list of 4 scalars (types: str,int,int,str) containing
#    CHR,START,END,EXON_ID
# -> 'samples' is the list of sampleIDs (ie strings)
# -> 'counts' is a numpy 2D-array of uint32 of size nbWindows * nbSamples
def parseCountsFile(countsFile):
    try:
        npzCounts = numpy.load(countsFile)
    except Exception as e:
        logger.error("Opening provided countsFile %s: %s", countsFile, e)
        raise Exception('cannot open countsFile')

    # samples
    samples = (npzCounts['samples']).tolist()

    # (pseudo-)exons
    genomicWindows = exonsFromNdarray(npzCounts['exons'])

    # counts
    counts = npzCounts['counts']

    npzCounts.close()
    return(genomicWindows, samples, counts)


#################################################
# convertExon:
# Arg: a list of 4 strings corresponding to an exon (CHR,START,END,EXON_ID but
# with START and END as strings)
# Return: a similar list where START and END are converted to ints
def convertExon(eas):
    return([eas[0], int(eas[1]), int(eas[2]), eas[3]])


#################################################
# exonsFromNdarray:
# Arg: a 2D-array of strings representing genomicWindows (exons and intergenic
# pseudo-exons), as obtained via numpy.load()
# Return: the same data but structured as returned by processBed, ie a list of
# (pseudo-)exons, where each is a list of 4 scalars (types: str,int,int,str)
# containing CHR,START,END,EXON_ID
def exonsFromNdarray(exonsNP):
    return(list(map(convertExon, exonsNP.tolist())))
