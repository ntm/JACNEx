###############################################################################################
######################################## MAGE-CNV step 2: Normalisation & clustering ##########
###############################################################################################
# Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and forms the reference 
# clusters for the call. 
# Prints results in a folder defined by the user. 
# See usage for more details.
###############################################################################################

import sys
import getopt
import os
import numpy as np
import time
import logging

import matplotlib.pyplot

# different scipy submodules are used for the application of hierachical clustering 
import scipy.cluster.hierarchy 
import scipy.spatial.distance  

# import sklearn submodule for  Kmeans calculation
import sklearn.cluster

# prevent matplotlib DEBUG messages filling the logs when we are in DEBUG loglevel
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

# prevent numba DEBUG messages filling the logs when we are in DEBUG loglevel
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# !!!! definition of the logger here as the functions are not yet modularised (will change) 
# configure logging, sub-modules will inherit this config
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
# set up logger: we want scriptName rather than 'root'
logger = logging.getLogger(os.path.basename(sys.argv[0]))

################################################################################################
################################ Modules #######################################################
################################################################################################
# parse a pre-existing counts file
from mageCNV.countsFile import parseCountsFile 

# normalizes a fragment counting array to FPM. 
from mageCNV.normalisation import FPMNormalisation 

################################################################################################
################################ Functions #####################################################
################################################################################################
##############################################
# From a list of exons, identification of gonosomes and genders.
# These gonosomes are predefined and limited to the X,Y,Z,W chromosomes present
# in most species (mammals, birds, fish, reptiles).
# The number of genders is therefore limited to 2, i.e. Male and Female
# Arg:import scipy.cluster.hierarchy
# Returns a tuple (gonoIndexDict, gendersInfos), each are created here:
# -> 'gonoIndexDict' is a dictionary where key=GonosomeID(e.g 'chrX')[str], 
# value=list of gonosome exon index [int]. It's populated from the exons list. 
# -> 'gendersInfos' is a str list of lists, contains informations for the gender
# identification, ie ["gender identifier","particular chromosome"].
# The indexes of the different lists are important:
# index 0: gender carrying a unique gonosome not present in the other gender (e.g. human => M:XY)
# index 1: gender carrying two copies of the same gonosome (e.g. human => F:XX) 

def getGenderInfos(exons):
    # pre-defined list of gonosomes
    # the order of the identifiers is needed to more easily identify the 
    # combinations of chromosomes. 
    # combinations: X with Y and Z with W + alphabetical order
    gonoChromList = ["X", "Y", "W", "Z"]
    # reading the first line of "exons" for add "chr" to 'gonoChromList' in case 
    # of this line have it
    if (exons[0][0].startswith("chr")):
        gonoChromList = ["chr" + letter for letter in gonoChromList]
    
    # for each exon in 'gonoChromList', add exon index in int list value for the 
    # correspondant gonosome identifier (str key). 
    gonoIndexDict=dict()
    for exonIndex in range(len(exons)):
        if (exons[exonIndex][0] in gonoChromList):
            if (exons[exonIndex][0] in gonoIndexDict):
                gonoIndexDict[exons[exonIndex][0]].append(exonIndex)
            # initialization of a key, importance of defining the value as a list 
            # to allow filling with the next indices.
            else:
                gonoIndexDict[exons[exonIndex][0]] = [exonIndex]
        # exon in an autosome
        # no process next
        else:
            continue
            
    # the dictionary keys may not be sorted alphabetically
    # needed to compare with gonoChromList      
    sortKeyGonoList = list(gonoIndexDict.keys())
    sortKeyGonoList.sort()
    if (sortKeyGonoList==gonoChromList[:2]):
        # Human case:
        # index 0 => Male with unique gonosome chrY
        # index 1 => Female with 2 chrX 
        genderInfoList = [["M",sortKeyGonoList[1]],["F",sortKeyGonoList[0]]]
    elif (sortKeyGonoList==gonoChromList[2:]):
        # Reptile case:
        # index 0 => Female with unique gonosome chrW
        # index 1 => Male with 2 chrZ 
        genderInfoList = [["F",sortKeyGonoList[0]],["M",sortKeyGonoList[1]]]
    else:
        logger.error("No X, Y, Z, W gonosomes are present in the exon list.\n \
        Please check that the exon file initially a BED file matches the gonosomes processed here.")
        sys.exit(1) 
    return(gonoIndexDict, genderInfoList)

##############################################
# clusterBuilds :
# Group samples with similar coverage profiles (FPM standardised counts).
# Use absolute pearson correlation and hierarchical clustering.
# Args:
#   -FPMArray is a float numpy array, dim = NbExons x NbSOIs 
#   -SOIs is the str sample of interest name list 
#   -minSampleInCluster is an int variable, defining the minimal sample number to validate a cluster
#   -minLinks is a float variable, it's the minimal distance tolerated for build clusters 
#   (advice:run the script once with the graphical output to deduce this threshold as specific to the data)
#   -figure: is a boolean: True or false to generate a figure
#   -outputFile: is a full path (+ file name) for saving a dendogram

#Return :
# -resClustering: a list of list with different columns typing, it's the clustering results,
# the list indexes are ordered according to the SOIsIndex
#  dim= NbSOIs*4 columns: 
# 1) sampleName [str], 
# 2) clusterID [int], 
# 3) controlledBy [ints list], 
# 4) validitySamps [int], boolean 0: dubious sample and 1: valid sample,

# -[optionally] a png showing a dendogram

def clusterBuilds(FPMArray, SOIs, minSampleInCluster, minLinks, figure, outputFile):
    #####################################################
    # Correlation:
    ##################
    #corrcoef return Pearson product-moment correlation coefficients.
    #rowvar=False parameter allows you to correlate the columns=sample and not the rows[exons].
    correlation = np.round(np.corrcoef(FPMArray,rowvar=False),2) #correlation value decimal < -10² = tricky to interpret
    ######################################################
    # Distance: 
    ##################
    # Euclidean distance (classical method) not used
    # Absolute correlation distance is unlikely to be a sensible distance when 
    # clustering samples. (1 - r where r is the Pearson correlation)
    dissimilarity = 1 - abs(correlation) 
    # Complete linkage, which is the more popular method,
    # Distances between the most dissimilar members for each pair of clusters are calculated
    # and then clusters are merged based on the shortest distance
    # f=max(d(x,y)) ; x is a sample in one cluster and y is a sample in the other.
    # DEV WARNING : probably to be changed to a more sophisticated method like Ward
    # scipy.squareform is necessary to convert a vector-form distance vector to a square-form 
    # distance matrix readable by scipy.linkage
    samplesLinks = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'complete')
    
    ######################################################
    # Clustering :
    ##################
    # Iteration on decreasing distance values every 0.05.
    # Cluster formation if its size is greater than minSamplesInCluster.
    # Formation of the most correlated clusters first, they will be used as a control
    # for the clusters obtained at the next lower distance value.
    # For the last iteration (lowest distance), if the number of clusters is not 
    # sufficient, a new cluster is created, but return a warning message for the clusterID and 
    # an invalidity status for the samples concerned.
        
    # Accumulators:
    # Attribute cluster identifier [int]
    clusterCounter = 0
    
    # To fill
    # a list of list with different types of arguments [str,int,intsList,int]
    resClustering=[0]*len(SOIs)
    # a dictionary with the correpondence between control cluster (key: int) 
    # and cluster to be controlled (value: int list)
    controlsDict={}

    # Distances range definition 
    # add 0.01 to minLinks as np.arrange(start,stop,step) don't conciderate stop in range 
    distanceStep = 0.05
    linksRange = np.arange(distanceStep,minLinks+0.01,distanceStep)
 
    for selectLinks in linksRange:
        # fcluster : Form flat clusters from the hierarchical clustering defined by 
        # the given linkage matrix.
        # Return An 1D array, dim= sampleIndex[clusterNb]
        clusterFormedList = scipy.cluster.hierarchy.fcluster(samplesLinks, selectLinks, criterion='distance')

        # a list of the unique identifiers of the clusters [int]
        uniqueClusterID = np.unique(clusterFormedList)
        
        ######################
        # Cluster construction
        # all clusters obtained for a distance value (selectLinks) are processed
        for clusterIndex in range(len(uniqueClusterID)):
            # list of sample indexes associated with this cluster
            SOIIndexInCluster, = list(np.where(clusterFormedList == uniqueClusterID[clusterIndex]))
            
            #####################
            # Cluster selection criterion, enough samples number in cluster
            if (len(SOIIndexInCluster)>=minSampleInCluster):
                # New samplesIndexes to fill in resClustering list
                SOIIndexToAddList = [index for index,value in enumerate(resClustering) 
                                   if (index in SOIIndexInCluster and value==0)]
                
                # Create a new cluster if new samples are presents                                    
                if (len(SOIIndexToAddList)!=0):
                    clusterCounter += 1
                    # selection of samples indexes already present in older clusters => controls clusters 
                    IndexToGetClustInfo = set(SOIIndexInCluster)-set(SOIIndexToAddList)
                    # list containing all unique clusterID as control for this new cluster [int]
                    listCtrlClust = set([value[1] for index,value in enumerate(resClustering) 
                                       if (index in IndexToGetClustInfo)])
                    # Filling resClustering for new samples
                    for SOIIndex in SOIIndexToAddList:
                        resClustering[SOIIndex]=[SOIs[SOIIndex],clusterCounter,list(listCtrlClust),1]
                    # Filling controlsDict if new cluster is controlled by previous cluster(s)
                    for ctrl in listCtrlClust:
                        if ctrl in controlsDict:
                            tmpList=controlsDict[ctrl]
                            tmpList.append(clusterCounter)
                            controlsDict[ctrl]=tmpList
                        else:
                            controlsDict[ctrl]=[clusterCounter]
                    
                # probably a previous formed cluster, same sample composition, no analysis is performed
                else:
                    continue
            # not enough samples number in cluster              
            else: 
                #####################
                # Case where it's the last distance threshold and the samples have never been seen
                # New cluster creation with dubious samples for calling step 
                if (selectLinks==linksRange[-1]):
                    clusterCounter += 1
                    logger.warning("Creation of cluster n°%s with insufficient numbers %s with low correlation %s",
                    clusterCounter,str(len(SOIIndexInCluster)),str(selectLinks))
                    for SOIIndex in SOIIndexInCluster:
                        resClustering[SOIIndex] = [SOIs[SOIIndex],clusterCounter,list(),0]
                else:
                    continue  
    # case we wish to obtain the graphical representation
    if figure:
        try:
            clusterDendogramsPrivate(resClustering, controlsDict,clusterCounter, samplesLinks, outputFile)
        except Exception as e: 
            logger.error("clusterDendograms failed - %s", e)
            sys.exit(1)

    return(resClustering)

########################################
# visualisation of clustering results
# Args:
# -resClustering: a list of list with dim: NbSOIs*4 columns 
# ["sampleName"[str], "ClusterID"[int], "controlsClustID"[int], "ValiditySamps"[int]]
# -maxClust: a int variable for maximum clusterNb obtained after clustering
# -controlsDict: a dictionary key:controlsCluster[int], value:list of clusterID to Control[int]  
# -sampleLinks: a 2D numpy array of floats, correspond to the linkage matrix, dim= NbSOIs-1*4. 
# -outputFile : full path to save the png 
# Return a png to the user-defined output folder

def clusterDendogramsPrivate(resClustering, controlsDict,maxClust, sampleLinks, outputFile):
    #initialization of a list of strings to complete
    labelsGp=[]

    # Fill labelsgp at each new sample/row in resClustering
    # Cluster numbers link with the marking index in the label 
    # 2 marking type: "*"= target cluster ,"-"= control cluster present in controlsDict
    for sIndex in range(len(resClustering)):
        # an empty str list of length of the maximum cluster found
        tmpLabels=["   "]*maxClust
        row=resClustering[sIndex] 
        # select group-related position in tmplabels
        #-1 because the group numbering starts at 1 and not 0
        indexCluster=row[1]-1
        # add symbol to say that this sample is a target 
        tmpLabels[indexCluster]=" * "
        # case the sample is part of a control group
        # add symbol to the index of groups to be controlled in tmpLabels
        if row[1] in controlsDict:
            for gpToCTRL in controlsDict[row[1]]:
                # symbol add to tmplabels for the current group index.
                tmpLabels[gpToCTRL-1]=" - "
        labelsGp.append("".join(tmpLabels))

    # dendogram plot
    matplotlib.pyplot.figure(figsize=(5,15),facecolor="white")
    matplotlib.pyplot.title("Complete linkage hierarchical clustering")
    dn1 = scipy.cluster.hierarchy.dendrogram(sampleLinks,orientation='left',
                                             labels=labelsGp, 
                                             color_threshold=0.05)
    
    matplotlib.pyplot.ylabel("Samples of interest")
    matplotlib.pyplot.xlabel("Absolute Pearson distance (1-r)")
    matplotlib.pyplot.savefig(outputFile, dpi=520, format="png", bbox_inches='tight')

######################################
# Gender and group predicted by Kmeans matching 
# calcul of normalized count ratios per gonosomes and kmeans group
# ratio = median (normalized count sums list for a specific gonosome and for all samples in a kmean group)
# Args:
#  -kmeans: an int list of groupID predicted by Kmeans ordered on SOIsIndex 
#  -countsNorm: a float 2D numpy array of normalized count, dim=NbExons*NbSOIs
#  -gonoIndexDict: a dictionary of correspondence between gonosomeID[key:str] and exonsIndex[value:int list] 
#  -genderInfoList: a list of list, dim=NbGender*2columns ([genderID, specificChr])
# Returns a list of genderID where the indices match the groupID formed by the Kmeans.
# e.g ["F","M"], KmeansGp 0 = Female and KmeansGp 1 = Male

def genderAttributionPrivate(kmeans, countsNorm,gonoIndexDict, genderInfoList):
    # initiate float variable, 
    # save the first count ratio for a given gonosome
    previousCount=None

    # first browse on gonosome names (limited to 2)
    for gonoID in gonoIndexDict.keys():
        # int list of the current gonosome exons indexes 
        gonoExonIndexL=gonoIndexDict[gonoID]
        
        # second browse on the kmean groups (only 2)
        for kmeanGroup in np.unique(kmeans.labels_):
            # int list corresponding to the sample indexes in the current Kmean group
            SOIsIndexKGpL,=list(np.where(kmeans.labels_ == kmeanGroup))
            
            #####################
            # selection of specifics normalized count data
            tmpArray=countsNorm[gonoExonIndexL,] # gonosome exons
            tmpArray=tmpArray[:,SOIsIndexKGpL] # Kmean group samples        
            
            #####################
            # ratio calcul (axis=0, sum all row/exons for each sample) 
            countRatio = np.median(np.sum(tmpArray,axis=0))

            # Keep gender names in variables
            g1 = genderInfoList[0][0] #e.g human g1="M"
            g2 = genderInfoList[1][0] #e.g human g2="F"
            
            #####################
            # filling two lists corresponding to the gender assignment condition
            # condition1L and condition2L same construction
            # 1D string list , dim=2 genderID (indexes corresponds to Kmeans groupID) 
            if (previousCount!=None):    
                # row order in genderInfoList is important
                # 0 : gender with specific gonosome not present in other gender
                # 1 : gender with 2 copies of same gonosome
                if gonoID == genderInfoList[0][1]: # e.g human => M:chrY
                    #####################
                    # condition assignment gender number 1:
                    # e.g human case group of M, the chrY ratio is expected 
                    # to be higher than the group of F (>10*)
                    countsx10=10*countRatio
                    condition1L=[""]*2
                    if previousCount>countsx10:
                        condition1L[kmeanGroup-1]=g1 #e.g human Kmeans gp0 = M
                        condition1L[kmeanGroup]=g2 #e.g human Kmeans gp1 = F
                    else:
                        condition1L[kmeanGroup-1]=g2 #e.g human Kmeans gp0 = F
                        condition1L[kmeanGroup]=g1 #e.g human Kmeans gp1 = M
                else: # e.g human => F:chrX
                    #####################
                    # condition assignment gender number 2:
                    # e.g human case group of F, the chrX ratio should be in 
                    # the range of 1.5*ratiochrXM to 3*ratiochrXM
                    countsx1half=3*countRatio/2
                    condition2L=[""]*2
                    if previousCount>countsx1half and previousCount<2*countsx1half:
                        condition2L[kmeanGroup-1]=g2
                        condition2L[kmeanGroup]=g1
                    else:
                        condition2L[kmeanGroup-1]=g1
                        condition2L[kmeanGroup]=g2
            
                # restart for the next gonosome
                previousCount=None
            else:
                # It's the first ratio calculated for the current gonosome => saved for comparison 
                # with the next ratio calcul
                previousCount=countRatio
            
            
    # predictions test for both conditions
    # the two lists do not agree => raise an error and quit the process.
    if condition1L!=condition1L:
        logger.error("The conditions of gender allocation are not in agreement.\n \
            condition n°1, one gender is characterised by a specific gonosome: %s \n \
                condition n°2 that the other gender is characterised by 2 same gonosome copies: %s ", condition1L, condition2L)
        sys.exit(1)
    return(condition1L)

################################################################################################
######################################## Main ##################################################
################################################################################################
def main():
    scriptName = os.path.basename(sys.argv[0])
    ##########################################
    # parse user-provided arguments
    # mandatory args
    countsFile = ""
    outFolder = ""
    ##########################################
    # optionnal arguments
    # default values fixed
    minSamples = 20
    minLinks = 0.25
    nogender = False
    figure = False

    usage = """\nCOMMAND SUMMARY:
Given a TSV of exon fragment counts, normalizes the counts (Fragment Per Million) and forms the reference 
clusters for the call. 
By default, separation of autosomes and gonosomes (chr accepted: X, Y, Z, W) for clustering, to avoid bias.
Results are printed to stdout folder:
- a TSV file format, describe the clustering results, dim = NbSOIs*8 columns:
    1) "sampleName": a string for sample of interest full name,
    2) "clusterID_A": an int for the cluster containing the sample for autosomes (A), 
    3) "controlledBy_A": an int list of clusterID controlling the current cluster for autosomes,
    4) "validitySamps_A": a boolean specifying if a sample is dubious(0) or not(1) for the calling step, for autosomes.
                          This score set to 0 in the case the cluster formed is validated and does not have a sufficient 
                          number of individuals.
    5) "genderPreds": a string "M"=Male or "F"=Female deduced by kmeans,
    6) "clusterID_G": an int for the cluster containing the sample for gonosomes (G), 
    7) "controlledBy_G":an int list of clusterID controlling the current cluster ,
    8) "validitySamps_G": a boolean specifying if a sample is dubious(0) or not(1) for the calling step, for gonosomes.
- one or more png's illustrating the clustering performed by dendograms. [optionnal]
    Legend : solid line = target clusters , thin line = control clusters
    The clusters appear in decreasing order of distance.

ARGUMENTS:
   --counts [str]: TSV file, first 4 columns hold the exon definitions, subsequent columns 
                   hold the fragment counts.
   --out[str]: pre-existing folder to save the output files
   --minSamples [int]: an integer indicating the minimum sample number to create a reference cluster for autosomes,
                  default : """+str(minSamples)+""".
   --minLinks [float]: a float indicating the minimal distance to considered for the hierarchical clustering,
                  default : """+str(minLinks)+""".   
   --nogender[optionnal]: no autosomes and gonosomes discrimination for clustering. 
                  output TSV : dim= NbSOIs*4 columns, ["sampleName", "clusterID", "controlledBy", "validitySamps"]
   --figure[optionnal]: make one or more dendograms that will be present in the output in png format."""+"\n"

    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],'h',
        ["help","counts=","out=","minSamples=","minLinks=","nogender","figure"])
    except getopt.GetoptError as e:
        sys.exit("ERROR : "+e.msg+".\n"+usage)

    for opt, value in opts:
        # sanity-check and store arguments
        if (opt in ('-h', '--help')):
            sys.stderr.write(usage)
            sys.exit(0)
        elif (opt in ("--counts")):
            countsFile = value
            if (not os.path.isfile(countsFile)):
                sys.exit("ERROR : countsFile "+countsFile+" doesn't exist. Try "+scriptName+" --help.\n")
        elif (opt in ("--out")):
            outFolder = value
            if (not os.path.isdir(outFolder)):
                sys.exit("ERROR : outFolder "+outFolder+" doesn't exist. Try "+scriptName+" --help.\n")
        elif (opt in ("--minSamples")):
            try:
                minSamples = np.int(value)
            except Exception as e:
                logger.error("Conversion of 'minSamples' value to int failed : %s", e)
                sys.exit(1)
        elif (opt in ("--minLinks")):
            try:
                minLinks = np.float(value)
            except Exception as e:
                logger.error("Conversion of 'minLinks' value to int failed : %s", e)
                sys.exit(1)
        elif (opt in ("--nogender")):
            nogender = True
        elif (opt in ("--figure")):
            figure = True
        else:
            sys.exit("ERROR : unhandled option "+opt+".\n")

    #####################################################
    # Check that the mandatory parameter is present
    if (countsFile == ""):
        sys.exit("ERROR : You must use --counts.\n"+usage)

    ######################################################
    # args seem OK, start working
    logger.info("starting to work")
    startTime = time.time()

    # parse counts from TSV to obtain :
    # - a list of exons same as returned by processBed, ie each
    #    exon is a lists of 4 scalars (types: str,int,int,str) containing CHR,START,END,EXON_ID
    #    copied from the first 4 columns of countsFile, in the same order
    # - the list of sampleIDs (ie strings) copied from countsFile's header
    # - an int numpy array, dim = len(exons) x len(samples)
    try:
        exons, SOIs, countsArray = parseCountsFile(countsFile)
    except Exception as e:
        logger.error("parseCountsFile failed - %s", e)
        sys.exit(1)

    thisTime = time.time()
    logger.debug("Done parsing countsFile, in %.2f s", thisTime-startTime)
    startTime = thisTime
    
    #####################################################
    # Normalisation:
    ##################  
    # allocate countsNorm and populate it with normalised counts of countsArray
    # same dimension for arrays in input/output: exonIndex*sampleIndex
    # Fragment Per Million normalisation
    # NormCountOneExonForOneSample=(FragsOneExonOneSample*1x10^6)/(TotalFragsAllExonsOneSample)
    try :
        countsNorm = FPMNormalisation(countsArray)
    except Exception as e:
        logger.error("FPMNormalisation failed - %s", e)
        sys.exit(1)
    thisTime = time.time()
    logger.debug("Done FPM normalisation, in %.2f s", thisTime-startTime)
    startTime = thisTime

    #####################################################
    # Clustering:
    ####################
    # case where no discrimination between autosomes and gonosomes is made
    # direct application of the clustering algorithm
    if nogender:
        try: 
            outputFile=os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_FullChrom.png")
            resClustering = clusterBuilds(countsNorm, SOIs, minSamples, minLinks, figure, outputFile)
        except Exception as e: 
            logger.error("clusterBuilding failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.debug("Done clusterisation in %.2f s", thisTime-startTime)
        startTime = thisTime
    # cases where discrimination is made 
    # avoid grouping Male with Female which leads to dubious CNV calls 
    else:
        #identification of gender and keep exons indexes carried by gonosomes
        try: 
            gonoIndexDict, genderInfoList = getGenderInfos(exons)
        except Exception as e: 
            logger.error("getGenderInfos failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.debug("Done get gender informations in %.2f s", thisTime-startTime)
        startTime = thisTime

        #division of normalized count data according to autosomal or gonosomal exons
        #flat gonosome index list
        gonoIndex = np.unique([item for sublist in list(gonoIndexDict.values()) for item in sublist]) 
        autosomesFPM = np.delete(countsNorm,gonoIndex,axis=0)
        gonosomesFPM = np.take(countsNorm,gonoIndex,axis=0)

        #####################################################
        # Get Autosomes Clusters
        ##################
        # Application of hierarchical clustering on autosome count data to obtain :
        # - a 2D numpy array with different columns typing, extract clustering results,
        #       dim= NbSOIs*4. columns: ,1) SOIs Names [str], 2)clusterID [int], 
        #       3)clusterIDToControl [str], 4) Sample validity for calling [int] 

        logger.info("### Samples clustering on normalised counts for autosomes")
        try :
            outputFile=os.path.join(outFolder,"Dendogram_"+str(len(SOIs))+"Samps_Autosomes.png")
            logger.info("Normally output File %s, %s",outputFile,outFolder)
            resClusteringAutosomes = clusterBuilds(autosomesFPM, SOIs, minSamples, minLinks, figure, outputFile)
        except Exception as e:
            logger.error("clusterBuilds for autosomes failed - %s", e)
            sys.exit(1)
        thisTime = time.time()
        logger.info("Done samples clustering for autosomes : in %.2f s", thisTime-startTime)
        startTime = thisTime

        #####################################################
        # Get Gonosomes Clusters
        ##################
        # Different treatment
        # It is necessary to have the gender genotype information
        # But without appriori a Kmeans can be made to split the data on gender number
        logger.info("### Samples clustering on normalised counts of gonosomes")
        kmeans =sklearn.cluster.KMeans(n_clusters=len(genderInfosDict.keys()), random_state=0).fit(gonosomesFPM.T)#transposition to consider the samples



if __name__ =='__main__':
    main()