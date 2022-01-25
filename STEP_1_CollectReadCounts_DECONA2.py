#############################################################################################################
######################################## STEP1 Collect Read Count DECONA2 ###################################
#############################################################################################################
#STEPS:
#   A) Getopt user argument (ARGV) recovery
#   B) Checking that the mandatory parameters are presents
#   C) Checking that the parameters actually exist and processing
#       concerns the creation of a list containing the bams paths
#       based on the selection of either --bams or --bams-from arguments.
#   D) Parsing exonic intervals bed
#       "processBed" function used to check the integrity of the file, padding the intervals
#        +-10 bp, sorting the positions on the 4 columns (1:CHR,2:START,3:END,4:Exon_ID)
#   E) Creating NCL for each chromosome
#       "exonDict" function used from the ncls module, creation of a dictionary :
#        key = chromosome , value : interval_tree
#   F) Parsing old counts file (.tsv) if exist else new count Dataframe creation 
#       "parseCountFile" function used to check the integrity of the file against the previously
#        generated bed file. It also allows to check if the pre-existing counts are in a correct format.
#   G) Definition of a loop for each BAM files and the reads counting.
#       "countFrags" function : 
#           -allows to sort alignments by BAM QNAME and to filter them on specific flags (see conditions in the script).
#            Realisation by samtool v1.9 (using htslib 1.9)
#           -also extracts the information for each unique QNAME. The information on the end position and the length of
#           the alignments are not present the "AliLengthOnRef" function retrieves them.
#           -the Qname informations is sent to the "Qname2ExonCount" function to perform the fragment count. 
#           -A check of the results is also performed.
#   This step completes the count dataframe.
#   Once all samples have been processed, this dataframe is saved.

#############################################################################################################
################################ Loading of the modules required for processing #############################
#############################################################################################################
# 1) Python Modules
import sys
import getopt
import logging
import os
import pandas as pd # dataframe objects
import re
# nested containment lists, similar to interval trees but faster (https://github.com/biocore-ntnu/ncls)
from ncls import NCLS
import subprocess # run samtools
import tempfile

#####################################################################################################
################################ Logging Definition #################################################
#####################################################################################################
# set up logger
logger=logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.DEBUG)
# create console handler for STDERR
stderr = logging.StreamHandler(sys.stderr)
stderr.setLevel(logging.DEBUG)
#create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s',
                                '%Y-%m-%d %H:%M:%S')
#add formatter to stderr handler
stderr.setFormatter(formatter)
#add stderr handler to logger
logger.addHandler(stderr)

#####################################################################################################
################################ Functions ##########################################################
#####################################################################################################

####################################################
#Exon intervals file parsing and preparing.
# Input : bedFile == a bed file (with path), possibly gzipped, containing exon definitions
#         formatted as CHR START END EXON_ID
#
# Output : returns the data as a pandas dataframe with column headers CHR START END EXON_ID,
#          and where:
# - a 10bp padding is added to exon coordinates (ie -10 for START and +10 for END)
# - exons are sorted by CHR, then START, then END, then EXON_ID
def processBed(PathBedToCheck):
    bedname=os.path.basename(PathBedToCheck)
    if not os.path.isfile(PathBedToCheck):
        logger.error("BED file %s doesn't exist.\n", PathBedToCheck)
        sys.exit(1)
    try:
        BedToCheck=pd.read_table(PathBedToCheck, header=None, sep="\t")
        # compression == 'infer' by default => auto-works whether bedFile is gzipped or not
    except Exception as e:
        logger.error("error parsing BED file %s: %s", bedname, e)
        sys.exit(1)
   #####################
    #I): Sanity Check
    #####################
    if len(BedToCheck.columns) != 4:
        logger.error("BED file %s should be a 4-column TSV (CHR, START, END, EXON_ID) but it has %i columns\n",
                     bedname, len(BedToCheck.columns))
        sys.exit(1)
    BedToCheck.columns=["CHR","START","END","EXON_ID"]
    #######################
    #CHR column
    if (BedToCheck["CHR"].dtype=="O"): # CHR is a str
        if BedToCheck["CHR"].str.startswith('chr').all:
            BedToCheck['CHR_NUM'] = BedToCheck['CHR'].replace(regex=r'^chr(\w+)$', value=r'\1')
        else:
            BedToCheck['CHR_NUM']=BedToCheck['CHR']
    else:
        logger.error("In BED file %s, first column 'CHR' should be a string but pandas sees it as %s\n",
                     bedname, BedToCheck["CHR"].dtype)
        sys.exit(1)
    #######################
    #Start and End column
    if (BedToCheck["START"].dtype=="int") and (BedToCheck["END"].dtype=="int"):
        if (len(BedToCheck[BedToCheck.START<0])>0) or (len(BedToCheck[BedToCheck.END<0])>0):
            logger.error("In BED file %s, columns 2 and/or 3 contain negative values", bedname)
            sys.exit(1)
    else:
        logger.error("In BED file %s, columns 2-3 'START'-'END' should be ints but pandas sees them as %s - %s\n",
                     bedname, BedToCheck["START"].dtype, BedToCheck["END"].dtype)
        sys.exit(1)
    #######################
    #transcript_id_exon_number column
    if not (BedToCheck["EXON_ID"].dtype=="O"):
        logger.error("In BED file %s, 4th column 'EXON_ID' should be a string but pandas sees it as %s\n",
                     bedname, BedToCheck["EXON_ID"].dtype)
        sys.exit(1)
    # EXON_IDs must be unique
    if len(BedToCheck["EXON_ID"].unique()) != len(BedToCheck["EXON_ID"]):
        logger.error("In BED file %s, each line must have a unique EXON_ID (4th column)\n", bedname)
        sys.exit(1)

    #####################
    #II): Padding and sorting
    #####################
    # pad coordinates with +-10bp
    BedToCheck['START'] -= 10
    BedToCheck['END'] += 10

    # replace X Y M/MT (if present) by max(CHR)+1,+2,+3
    # maxChr must be the max among the int values in CHR_NUM column...
    maxChr = int(pd.to_numeric(BedToCheck['CHR_NUM'],errors="coerce").max(skipna=True))
    BedToCheck["CHR_NUM"]=BedToCheck["CHR_NUM"].replace(
        {'X': maxChr+1,
         'Y': maxChr+2,
         'M': maxChr+3,
         'MT': maxChr+3})

    # convert type of CHR_NUM to int and catch any errors
    try:
        BedToCheck['CHR_NUM'] = BedToCheck['CHR_NUM'].astype(int)
    except Exception as e:
        logger.error("While parsing BED file, failed converting CHR_NUM to int: %s", e)
        sys.exit(1)
    # sort by CHR_NUM, then START, then END, then EXON_ID
    exons= BedToCheck.sort_values(by=['CHR_NUM','START','END','EXON_ID'])
    # delete the temp column, and return result
    exons.drop(['CHR_NUM'], axis=1, inplace=True)    
    return(exons)

####################################################
#Create nested containment lists (similar to interval trees but faster), one per
# chromosome, storing the exons
#Input : dataframe of exons, as returned by processBed
#Output: dictionary(hash): key=chr, value=NCL
def createExonDict(exons):
    exonDict={}
    listCHR=list(exons.CHR.unique())
    for chr in listCHR:
        exonsOnChr=exons.loc[exons["CHR"]==chr]
        ncls = NCLS(exonsOnChr["START"], exonsOnChr["END"], exonsOnChr.index)
        exonDict[chr]=ncls
    return(exonDict)

#################################################
# parseCountFile:
#Input:
#   - countFile is a tsv file (with path), including column titles, as
#     specified previously
#   - exons is a dataframe holding exon definitions, padded and sorted,
#     as returned by processBed
#
#-> Parse countFile into a dataframe (will be returned)
#-> Check that the first 4 columns are identical to exons,
#    otherwise die with an error.
#-> check that the samples counts columns are [int] type.
# 
#Output: returns the Frag count results as a pandas dataframe with column 
# headers CHR START END EXON_ID,sampleName*n

def parseCountFile(countFile, exons):
    try:
        counts=pd.read_table(countFile,sep="\t")
    except Exception as e:
        logger.error("Parsing provided countFile %s: %s", countFile, e)
        sys.exit(1)
    if (len(counts)==len(exons)): # lines number comparison  
        #Type Check 
        if not (counts.dtypes["CHR"]=="O" and counts.dtypes["EXON_ID"]=="O"):
            logger.error("One or both of the 'CHR' and 'EXON_ID' columns are not in the correct format. Please check it.\n"
                        +"The column must be a python object [str]")
            sys.exit(1)
        elif not (counts.dtypes["START"]=="int" and counts.dtypes["END"]=="int"):
            logger.error("One or both of the 'START' and 'END' columns are not in the correct format. Please check it.\n"
                        +"The columns must contain integers.")
            sys.exit(1)
        #Check if data are identical
        elif not (counts['CHR'].isin(exons['CHR']).value_counts())[True]==len(exons):
            logger.error("'CHR' column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        elif not (counts['START'].isin(exons['START']).value_counts())[True]==len(exons):
            logger.error("'START' column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        elif not (counts['END'].isin(exons['END']).value_counts())[True]==len(exons):
            logger.error("END column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        elif not (counts['EXON_ID'].isin(exons['EXON_ID']).value_counts())[True]==len(exons):
            logger.error("'EXON_ID' column in counts dataframe isn't identical to those in the exonic interval file. Please check it.")
            sys.exit(1)
        else: 
            #check that the old samples columns data is in [int] format.
            namesSampleToCheck=[]
            for columnIndex in range(5,len(counts.columns)):
                if not counts.iloc[:,columnIndex].dtypes=="int":
                    namesSampleToCheck.append(counts.iloc[:,columnIndex].columns)
            if len(namesSampleToCheck)>0:
                logger.error("Columns in %s, sample(s) %s are not in [int] format.\n"+
                "Please check and correct these before trying again.", countFile,(",".join(namesSampleToCheck)))
                sys.exit(1)
    else:
        logger.error("Old counts file %s doesn't have the same lines number as the exonic interval file.\n"+
        "Transcriptome version has probably changed.\n"+
        "In this case the whole samples set re-analysis must be done.\n"+
        "Do not set the --counts option.",countFile)
        sys.exit(1)
    return(counts)

####################################################
# countFrags :
# Count the fragments in bamFile that overlap each exon described in exonDict.
# Arguments:
#   - a bam file (with path)
#   - the dictionary of exon NCLs
#   - the total number of exons
#   - a tmp dir with fast RW access and enough space for samtools sort
#   - the number of cpu threads that samtools can use
#
# Returns a vector containing the fragment counts, in the same order as in the bed
# file used to create the NCLs.
# Raises an exception if something goes wrong
def countFrags(bamFile, exonDict, nbOfExons, processTmpDir, num_threads):
    with tempfile.TemporaryDirectory(dir=processTmpDir) as SampleTmpDir:
        ############################################
        # I] preprocessing:
        # Our algorithm needs to parse the alignements sorted by qname,
        # "samtools sort -n" allows this
        cmd1 ="samtools sort -n "+bamFile+" -@ "+str(num_threads)+" -T "+SampleTmpDir+" -O sam"
        p1 = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)
        
        # We can also immediately filter out poorly mapped (low MAPQ) or low quality
        # reads (based on the SAM flags) with samtools view:
        # -q sets the minimum mapping quality;
        # -F excludes alignements where any of the specified bits is set, we want to filter out:
        #   4 0x4 Read Unmapped
        #   256 0x80 not primary alignment
        #   512 0x200 Read fails platform/vendor quality checks
        #   1024 0x400 Read is PCR or optical duplicate
        # Total Flag decimal = 1796
        # For more details on FLAGs read the SAM spec: http://samtools.github.io/hts-specs/
        cmd2 ="samtools view -q 20 -F 1796"
        p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout, stdout=subprocess.PIPE, universal_newlines=True)

        # III] initialization
        # list containing the number of fragments overlapping each exon, indexes
        # correspond to the labels in the NCL
        vecExonCount=[0]*nbOfExons

        #Process control : skip mention corresponds to the Qnames removal for fragments counts
        #Aligments = alignements count, QN= qnames counts , Frag= fragments counts
        keys=["AlignmentsInBam",
            "AlignmentsOnMainChr",
            "QNProcessed",  #sum of all folowing keys => control
            "QNAliOnDiffChrSkip",
            "QNSameReadDiffStrandSkip",
            "QNSingleReadSkip",
            "QNNbAliR&FBetween1-3", #sum of the following 7 keys => control
            "QN1F&1R",
            "QN2F&1R_2FOverlapSkip",
            "QN2F&1R_SA",
            "QN1F&2R_2ROverlapSkip",
            "QN1F&2R_SA",
            "QNAliGapLength>1000bpSkip",
            "QNAliBackToBackSkip",
            "QNAli2F&2R_SA", #sum of the following 2 keys => control
            "QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip",
            "QNAliUnsuitableCombination_aliRorF>3Skip",
            "FragOverlapOnTargetInterval", # equal to the sum of VecExonCount results
            "QNNoOverlapOnTargetIntervalSkip",
            "QNOverlapOnTargetInterval"]
        dictStatCount={ i : 0 for i in keys }

        #Variables to be processed for each Qname's alignment.
        qchrom="" # unique
        qname="" # unique
        qstartR=[] #complete list if same Qname and same strand.
        qstartF=[]
        qendR=[]
        qendF=[]
        qReads=0 # two bits : 01 First read was seen , 10 Second read was seen
        # qFirstOnForward==1 if the first-in-pair read of this qname is on the
        # forward reference strand, 0 if it's on the reverse strand, -1 if we don't yet know
        qFirstOnForward=-1
        # qBad==True if qname contains alignments on differents chromosomes,
        # or if some of it's alis disagree regarding the strand on which the
        # first/last read-in-pair aligns
        qBad=False

        ############################################
        # IV] Regular expression definition
        mainChr=re.compile("^chr[\dXYM]\d?$")
        ############################################
        # V] Function Main loop
        #Browse the file sorted on the qnames and having undergone the appropriate filters
        for line in p2.stdout:
            line=line.rstrip('\r\n')
            align=line.split('\t')
            dictStatCount["AlignmentsInBam"]+=1

            #################################################################################################
            #A] Selection of information for the current alignment only if it is on main chr
            if not mainChr.match(align[2]): continue #TODO make this work for different naming conventions
            dictStatCount["AlignmentsOnMainChr"]+=1 #Only alignments on the main chromosomes are treated

            #################################################################################################
            #B] If we are done with previous qname: process it and reset accumulators
            if (qname!=align[0]) and (qname!=""):  # align[0] is the qname
                dictStatCount["QNProcessed"]+=1
                if not qBad:
                    Qname2ExonCount(qchrom,qstartF,qendF,qstartR,qendR,qReads,exonDict,vecExonCount,
                                    dictStatCount)
                qchrom=""
                qname=""
                qstartR=[]
                qstartF=[]
                qendR=[]
                qendF=[]
                qReads=0
                qFirstOnForward=-1
                qBad=False
            elif qBad: #same qname as previous ali but we know it's bad -> skip
                continue

            #################################################################################################
            #C] Either we're in the same qname and it's not bad, or we changed qname -> in both
            # cases update accumulators with current line
            if qname=="" :
                qname=align[0]

            # align[2] == chrom
            if qchrom=="":
                qchrom=align[2]
            elif qchrom!=align[2]:
                qBad=True
                dictStatCount["QNAliOnDiffChrSkip"]+=1
                continue
            # else same chrom, don't modify qchrom

            #Retrieving flags for STRAND and first/second read
            currentFlag=int(align[1])
            #flag 16 the alignment is on the reverse strand
            currentStrand="F"
            if currentFlag&16 :
                currentStrand="R"
            #flag 64 the alignment is first in pair, otherwise 128
            if currentFlag&64:
                qReads|=1
            else:
                qReads|=2

            #Processing for the positions lists for ali R and F
            currentStart=int(align[3])
            #Calculation of the CIGAR dependent 'end' position
            currentCigar=align[5]
            currentAliLength=AliLengthOnRef(currentCigar)
            currentEnd=currentStart+currentAliLength-1
            if currentStrand=="F":
                qstartF.append(currentStart)
                qendF.append(currentEnd)
            else:
                qstartR.append(currentStart)
                qendR.append(currentEnd)

            # currentFirstOnForward==1 if according to this ali, the first-in-pair read is on
            # the forward strand, 0 otherwise
            currentFirstOnForward=0
            if ((currentFlag&64) and (currentStrand=="F")) or ((currentFlag&128) and (currentStrand=="R")) :
                currentFirstOnForward=1
            if qFirstOnForward==-1:
                # first ali for this qname
                qFirstOnForward=currentFirstOnForward
            elif qFirstOnForward!=currentFirstOnForward:
                qBad=True
                dictStatCount["QNSameReadDiffStrandSkip"]+=1
                continue
            # else this ali agrees with previous alis for qname -> NOOP

        #################################################################################################
        #VI]  Process last Qname
        dictStatCount["QNProcessed"]+=1
        if not qBad:
            Qname2ExonCount(qchrom,qstartF,qendF,qstartR,qendR,qReads,exonDict,vecExonCount,
                            dictStatCount)

        ##################################################################################################
        # VII] wait for samtools to finish cleanly and check return codes
        if (p1.wait() != 0):
            logger.error("in countFrags, while processing %s, the 'samtools sort' subprocess returned %s",
                         bamFile, p1.returncode)
            raise Exception("samtools-sort failed")
        if (p2.wait() != 0):
            logger.error("in countFrags, while processing %s, the 'samtools view' subprocess returned %s",
                         bamFile, p2.returncode)
            raise Exception("samtools-view failed")

        ##################################################################################################
        #VIII] sanity checks
        NBTotalQname=(dictStatCount["QNAliOnDiffChrSkip"]+
                      dictStatCount["QNSameReadDiffStrandSkip"]+
                      dictStatCount["QNSingleReadSkip"]+
                      dictStatCount["QN1F&1R"]+
                      dictStatCount["QN2F&1R_2FOverlapSkip"]+
                      dictStatCount["QN2F&1R_SA"]+
                      dictStatCount["QN1F&2R_2ROverlapSkip"]+
                      dictStatCount["QN1F&2R_SA"]+
                      dictStatCount["QNAliGapLength>1000bpSkip"]+
                      dictStatCount["QNAliBackToBackSkip"]+
                      dictStatCount["QNAli2F&2R_SA"]+
                      dictStatCount["QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip"]+
                      dictStatCount["QNAliUnsuitableCombination_aliRorF>3Skip"])
        if ((NBTotalQname != dictStatCount["QNProcessed"]) or 
            (sum(vecExonCount) != dictStatCount["FragOverlapOnTargetInterval"])):
            statslist=dictStatCount.items()
            logger.error("Not all of %s's qnames were processed! BUG in code, FIXME! Stats: %s\n"+
                         "Nb total Qname : %s. Nb Qname overlaping exons: %s\n",
                         bamFile,statslist,NBTotalQname,sum(vecExonCount))
            raise Exception("sanity-check failed")

        # AOK, return counts
        return(vecExonCount)     
    

####################################################
# AliLengthOnRef :
#Input : a CIGAR string
#Output : span of the alignment on the reference sequence, ie number of bases
# consumed by the alignment on the reference
def AliLengthOnRef(CIGARAlign):
    length=0
    # only count CIGAR operations that consume the reference sequence, see CIGAR definition 
    # in SAM spec available here: https://samtools.github.io/hts-specs/
    match = re.findall(r"(\d+)[MDN=X]",CIGARAlign)
    for op in match:
        length+=int(op)
    return(length)

####################################################
# Qname2ExonCount :
# Given data representing all alignments for a single qname:
# - apply QC filters to ignore aberrant or unusual alignments / qnames;
# - identify the genomic positions that are putatively covered by the sequenced fragment,
#   either actually covered by a sequencing read or closely flanked by a pair of mate reads;
# - identify exons overlapped by the fragment, and increment their count.
# Inputs:
#   -chr variable [str]
#   -4 lists for F and R strand positions (start and end) [int]
#   -a bit variable informing about the read pair integrity 
#   (two bits : 01 First read was seen , 10 Second read was seen)
#   -the dictionary containing the NCLs for each chromosome
#   -list of fragment counts for each exon, appropriate counts will be incremented
#   -dictionary of sanity-checking counters
# Nothing is returned, this function just updates vecExonCount and dictStatCount
def Qname2ExonCount(chromString,startFList,endFList,startRList,endRList,readsBit,exonDict,vecExonCount,dictStatCount):
    Frag=[] #fragment(s) intervals
    #####################################################################
    ###I) DIFFERENTS CONDITIONS ESTABLISHMENT : for fragment detection
    #####################################################################
    ########################
    #-Removing Qnames containing only one read (possibly split into several alignments, mostly due to MAPQ filtering)
    if readsBit!=3:
        dictStatCount["QNSingleReadSkip"]+=1
        return

    elif (1<len(startFList+startRList)<=3): # only 1 or 2 ali on each strand but not 2 on each
        dictStatCount["QNNbAliR&FBetween1-3"]+=1
        if max(startFList)<min(endRList):# alignments are not back-to-back
            GapLength=min(startRList)-max(endFList)# gap length between the two alignments (negative if overlapping)
            if (GapLength<=1000):
                if (len(startFList)==1) and (len(startRList)==1):# one ali on each strand, whether SA or not doesn't matter
                    dictStatCount["QN1F&1R"]+=1
                    Frag=[min(startFList[0],startRList[0]),max(endFList[0],endRList[0])]
                elif (len(startFList)==2) and (len(startRList)==1):
                    if (min(startFList)<max(endFList)) and (min(endFList)>max(startFList)):
                        dictStatCount["QN2F&1R_2FOverlapSkip"]+=1
                        return # Qname deletion if the ali on the same strand overlap (SV ?!)
                    else:
                        dictStatCount["QN2F&1R_SA"]+=1
                        Frag=[min(startFList[0],max(startRList)),max(endFList[0],max(endRList)),
                              min(startFList),min(endFList)]
                elif (len(startFList)==1) and (len(startRList)==2):
                    if (min(startRList)<max(endRList)) and (min(endRList)>max(startRList)):
                        dictStatCount["QN1F&2R_2ROverlapSkip"]+=1
                        return # Qname deletion if the ali on the same strand overlap (SV ?!)
                    else:
                        dictStatCount["QN1F&2R_SA"]+=1
                        Frag=[min(startFList[0],min(startRList)),max(endFList[0],min(endRList)),
                              max(startRList),max(endRList)]
            else: # Skipped Qname, GapLength too large, big del !?
                dictStatCount["QNAliGapLength>1000bpSkip"]+=1 # TODO extract the associated Qnames for breakpoint detection.
                return
        else: # Skipped Qname, ali back-to-back (SV, dup ?!)
            dictStatCount["QNAliBackToBackSkip"]+=1
            return

    elif(len(startFList)==2) and (len(startRList)==2): # only 2F and 2 R combinations
        #Only one possibility to generate fragments in this case
        if ( (min(startFList)<min(endRList)) and (min(endFList)>min(startRList)) and
             (max(startFList)<max(endRList)) and (max(endFList)>max(startRList)) ):
            Frag=[min(min(startFList),min(startRList)),
                   max(min(endFList),min(endRList)),
                   min(max(startFList),max(startRList)),
                   max(max(endFList),max(endRList))]
            dictStatCount["QNAli2F&2R_SA"]+=1
        else: # The alignments describe SVs that are too close => Skipped Qname
            dictStatCount["QNAli2F&2R_F1&R1_F2&R2_NotOverlapSkip"]+=1
            return
    else: # At least 3 alignments on one strand => Skipped Qname
        dictStatCount["QNAliUnsuitableCombination_aliRorF>3Skip"]+=1
        return
    #####################################################################
    ###II)- FRAGMENT COUNTING FOR EXONIC REGIONS
    #####################################################################
    #Retrieving the corresponding NCL
    RefNCL=exonDict[chromString]

    for idx in list(range(0,(int(len(Frag)/2)))):
        SelectInterval=RefNCL.find_overlap(Frag[2*idx],Frag[2*idx+1])
        # how many overlapping intervals did we find
        overlaps=0
        for interval in SelectInterval:
            indexIntervalRef=int(interval[2])
            vecExonCount[indexIntervalRef]+=1
            dictStatCount["FragOverlapOnTargetInterval"]+=1
            overlaps += 1

        if overlaps==0:
            dictStatCount["QNNoOverlapOnTargetIntervalSkip"]+=1
            return #if the frag does not overlap a target interval => Skip
        else:
            dictStatCount["QNOverlapOnTargetInterval"]+=1

##############################################################################################################
######################################### Script Body ########################################################
##############################################################################################################
def main():
    scriptName=os.path.basename(sys.argv[0])
    ##########################################
    # A) Getopt user argument (ARGV) recovery
    bams=""
    bamsFrom=""
    bedFile=""
    # default setting ARGV 
    countFile=""
    tmpDir="/tmp/"
    threads=10 

    usage = """\nCOMMAND SUMMARY:
Given a BED of exons and one or more BAM files, count the number of sequenced fragments
from each BAM that overlap each exon (+- 10bp padding).
Results are printed to stdout in TSV format: first 4 columns hold the exon definitions after
padding, subsequent columns (one per BAM) hold the counts. If a pre-existing counts file produced
by this program with the same BED is provided (with --counts), its content is copied and counting
is only performed for the new BAM(s).
ARGUMENTS:
   --bams [str]: comma-separated list of BAM files
   --bams-from [str]: text file listing BAM files, one per line
   --bed [str]: BED file, possibly gzipped, containing exon definitions (format: 4-column 
                headerless tab-separated file, columns contain CHR START END EXON_ID)
   --counts [str] optional: pre-existing counts file produced by this program, content will be copied
   --tmp [str]: pre-existing dir for temp files, faster is better (eg tmpfs), default: """+tmpDir+"""
   --threads [int]: number of threads to allocate for samtools sort (default: """+str(threads)+"\n"

    
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","bams=","bams-from=","bed=","counts=","tmp=","threads="])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        #variables association with user parameters (ARGV)
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--bams"):
            bams=value     
        elif opt in ("--bams-from"):
            bamsFrom=value
        elif opt in ("--bed"):
            bedFile =value
        elif opt in ("--counts"):
            countFile=value
        elif opt in ("--tmp"):
            tmpDir=value
        elif opt in ("--threads"):
            threads=int(value)
        else:
            sys.exit("ERROR : Programming error. Unhandled option "+opt+".\n")

    #####################################################
    # B) Checking that the mandatory parameters are presents
    if (bams=="" and bamsFrom=="") or (bams!="" and bamsFrom!=""):
        sys.exit("ERROR : You must use either --bams or --bams-from but not both.\n"+usage)
    if bedFile=="":
        sys.exit("ERROR : You must use --bedFile.\n"+usage)

    #####################################################
    # C) Checking that the parameters actually exist and processing
    bamsToProcess=[]
    if bams!="":
        bamsToProcess=bams.split(",")
    
    elif bamsFrom!="":
        if not os.path.isfile(bamsFrom):
            sys.exit("ERROR : bams-from file "+bamsFrom+" doesn't exist. Try "+scriptName+" --help.\n")
        else:
            bamListFile=open(bamsFrom,"r")
            for line in bamListFile:
                line = line.rstrip('\n')
                bamsToProcess.append(line)
    else:
        sys.exit("ERROR : bams and bamsFile both empty, IMPOSSIBLE")
    #Check that all bams exist
    for b in bamsToProcess:
        if not os.path.isfile(b):
            sys.exit("ERROR : BAM "+b+" doesn't exist. Try "+scriptName+" --help.\n")

    if (countFile!="") and (not os.path.isfile(countFile)):
        sys.exit("ERROR : countFile "+countFile+" doesn't exist. Try "+scriptName+" --help.\n") 

    if not os.path.isdir(tmpDir):
        sys.exit("ERROR : tmp directory "+tmpDir+" doesn't exist. Try "+scriptName+" --help.\n")

    if (threads<=0):
        sys.exit("ERROR : number of threads "+str(threads)+" must be positive. Try "+scriptName+" --help.\n")

    ######################################################
    # D) Parsing exonic intervals bed
    exons=processBed(bedFile)
    nbOfExons=len(exons)

    ######################################################
    # E) Creating NCLs for each chromosome
    exonDict=createExonDict(exons)

    ############################################
    # F) Parsing old counts file (.tsv) if provided, else make a new dataframe
    if (countFile!=""):
        counts=parseCountFile(countFile,exons)
    else:
        counts=exons

    #####################################################
    # G) Process each BAM
    for bam in bamsToProcess:
        sampleName=os.path.basename(bam)
        sampleName=sampleName.replace(".bam","")
        logger.info('Sample being processed : %s', sampleName)

        if sampleName in list(counts.columns[4:]):
            logger.info('Sample %s already present in counts file, skipping it\n', sampleName)
            continue
        else:
            try:
                FragVec = countFrags(bam, exonDict, nbOfExons, tmpDir, threads)
            except Exception as e:
                logger.warning("Failed to count fragments for sample %s, skipping it - exception: %s\n",
                               sampleName, e)
                continue
            counts[sampleName]=FragVec

    counts.to_csv(sys.stdout,sep="\t", index=False)
      

if __name__ =='__main__':
    main()
