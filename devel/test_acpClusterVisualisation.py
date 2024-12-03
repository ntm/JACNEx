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
# Given a clusterFile produced by s2_clusterSamps.py and a countsFile produced by s1_countFrags.py,
# 2D and 3D interactive acp plotting
# See usage for details.
###############################################################################################
import getopt
import logging
import matplotlib.pyplot
import os
import sklearn.decomposition
import sys
# from mpl_toolkits.mplot3d import Axes3D

####### JACNEx modules
import clusterSamps.clustFile
import countFrags.countsFile
import countFrags.bed

# set up logger, using inherited config, in case we get called as a module
logger = logging.getLogger(__name__)

# prevent matplotlib flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)


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
    clusterFile = ""
    plotDir = ""
    acpTitle = ""

    usage = "NAME:\n" + scriptName + """\n
DESCRIPTION:
Given a clusterFile produced by s2_clusterSamps.py and a countsFile produced by s1_countFrags.py:
Generate 2D PCA plots and interactive 3D PCA plots.
Each visualization depicts the distribution of samples (points) and the clusters to which they belong (color).

ARGUMENTS:
   --counts [str] : metadata file in TSV format (with path)
   --clusts [str] : input clusterFile (with path)
   --plotDir [str] : directory where the plots are saved (with path)
   --title [str] : universal plot title
   -h , --help : display this help and exit\n"""

    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'h', ["help", "counts=", "clusts=", "plotDir=", "title="])
    except getopt.GetoptError as e:
        raise Exception(e.msg + ". Try " + scriptName + " --help")
    if len(args) != 0:
        raise Exception("bad extra arguments: " + ' '.join(args) + ". Try " + scriptName + " --help")

    for opt, value in opts:
        if opt in ('-h', '--help'):
            sys.stderr.write(usage)
            sys.exit(0)
        elif opt in ("--counts"):
            countsFile = value
        elif opt in ("--clusts"):
            clusterFile = value
        elif opt in ("--plotDir"):
            plotDir = value
        elif opt in ("--title"):
            acpTitle = value
        else:
            raise Exception("unhandled option " + opt)

    # Check args
    if countsFile == "":
        raise Exception("you must provide a counts file with --counts. Try " + scriptName + " --help.")
    elif (not os.path.isfile(countsFile)):
        raise Exception("countsFile " + countsFile + " doesn't exist.")

    if clusterFile == "":
        raise Exception("you must provide a clusterFile file with --clusters. Try " + scriptName + " --help")
    elif not os.path.isfile(clusterFile):
        raise Exception("clusterFile " + clusterFile + " doesn't exist")

    if acpTitle == "":
        raise Exception("you must provide a generique file title with --title. Try " + scriptName + " --help")

    # test plotdir last so we don't mkdir unless all other args are OK
    if not os.path.isdir(plotDir):
        try:
            os.mkdir(plotDir)
        except Exception as e:
            raise Exception("plotDir " + plotDir + " doesn't exist and can't be mkdir'd: " + str(e))

    # AOK, return everything that's needed
    return(countsFile, clusterFile, plotDir, acpTitle)


####################################################
# main function
def main(argv):
    (countsFile, clusterFile, plotDir, acpTitle) = parseArgs(argv)

    try:
        (samples, autosomeExons, gonosomeExons, intergenics, autosomeFPMs, gonosomeFPMs,
         intergenicFPMs) = countFrags.countsFile.parseAndNormalizeCounts(countsFile)
    except Exception as e:
        logger.error("parseAndNormalizeCounts failed for %s : %s", countsFile, str(e))
        raise Exception("parseAndNormalizeCounts failed")

    try:
        (clust2samps, samp2clusts, fitWith, clust2gender, clustIsValid) = clusterSamps.clustFile.parseClustsFile(clusterFile)
    except Exception as e:
        logger.error("error parsing clusterFile: %s", str(e))
        raise

    ### user test variables
    CHRtype = ["A", "G"]
    numComp = 10
    compToView = [0, 1, 2]
    annotations = ["grexome0555", "grexome0529"]

    for chr in range(len(CHRtype)):
        if CHRtype[chr] == "A":
            FPMtoProcess = autosomeFPMs
        elif CHRtype[chr] == "G":
            FPMtoProcess = gonosomeFPMs

        # PCA
        # we don't really want the smallest possible number of dimensions, try
        # smallish dims (must be < nbExons and < nbSamples)
        dim = min(numComp, FPMtoProcess.shape[0], FPMtoProcess.shape[1])
        samplesInPCAspace = sklearn.decomposition.PCA(n_components=dim).fit_transform(FPMtoProcess.T)
        # create a cluster IDs list
        clusters = list(clust2samps.keys())
        # Replace with our actual cluster IDs
        cluster_ids = [cluster_id for cluster_id in clusters if cluster_id.startswith(CHRtype[chr])]
        # Define a colormap based on the number of distributions
        distColor = matplotlib.pyplot.cm.get_cmap('tab20', len(cluster_ids))

        # Create a dictionary to map cluster IDs to colors
        cluster_colors = {}
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_colors[cluster_id] = distColor(idx)

        # Initialize an empty dictionary to store sample colors
        samples_colors = {}
        for samp in samp2clusts.keys():
            samples_colors[samp] = cluster_colors[samp2clusts[samp][chr]]

        # Generate list of colors for each sample (same order to samplesInPCAspace)
        listcolor = []
        for samp in range(len(samples)):
            listcolor.append(samples_colors[samples[samp]])

        for i in range(len(compToView)):
            for j in range(i + 1, len(compToView)):
                toSave = os.path.join(plotDir, acpTitle + "_" + CHRtype[chr] + "_Comp" + str(compToView[i]) + "vsComp" + str(compToView[j]) + ".pdf")
                ACP2D(samplesInPCAspace, compToView[i], compToView[j], listcolor, cluster_colors, samples, annotations, toSave)

        toSave = os.path.join(plotDir, acpTitle + "_" + CHRtype[chr])
        ACP3Danimation(samplesInPCAspace, compToView, listcolor, cluster_colors, samples, annotations, toSave)


###############
# ACP2D
# Generate a 2D scatter plot of PCA-transformed data with customizable colors and annotations.
#
# Args:
# - samplesInPCAspace (numpy array): PCA-transformed data points to be plotted.
# - Dim1Index (int): Index of the first dimension to be plotted on the x-axis.
# - Dim2Index (int): Index of the second dimension to be plotted on the y-axis.
# - sampColorList (list or array): colors corresponding to each data point.
# - cluster_colors (dict): mapping cluster labels to their corresponding colors.
# - samples (list): samples identifiers (same order of samplesInPCAspace rows)
# - annotations (list): user-defined sample list
# - toSave (str): Filename or path to save the generated plot.
#
# This function does not return any values.
# It generates a scatter plot and saves it as an image file specified by the toSave argument.
def ACP2D(samplesInPCAspace, Dim1Index, Dim2Index, sampColorList, cluster_colors, samples, annotations, toSave):
    matplotlib.pyplot.figure(figsize=(10, 10))
    matplotlib.pyplot.scatter(samplesInPCAspace[:, Dim1Index],
                              samplesInPCAspace[:, Dim2Index],
                              c=sampColorList)
    custom_lines = [matplotlib.pyplot.Line2D([], [], ls="", marker='.', mec='k', mfc=c,
                                             mew=.1, ms=20) for c in cluster_colors.values()]
    if len(annotations) > 0:
        for annot in annotations:
            matplotlib.pyplot.text(samplesInPCAspace[samples.index(annot), Dim1Index],
                                   samplesInPCAspace[samples.index(annot), Dim2Index],
                                   annot)

    matplotlib.pyplot.legend(custom_lines, [lt for lt in cluster_colors.keys()], loc='best')
    matplotlib.pyplot.xlabel("component " + str(Dim1Index))
    matplotlib.pyplot.ylabel("component " + str(Dim2Index))
    matplotlib.pyplot.savefig(toSave, dpi=75)
    matplotlib.pyplot.close()


##############
# ACP3D
# Generate a 3D scatter plot animation based on PCA-transformed data points.
#
# Args:
# - samplesInPCAspace (np.ndarray): PCA-transformed data points.
# - compToView (list): List of three component indices for visualization.
# - sampColorList (list): List of colors corresponding to each data point.
# - cluster_colors (dict): Dictionary mapping cluster labels to their corresponding colors.
# - samples (list): samples identifiers (same order of samplesInPCAspace rows)
# - annotations (list): user-defined sample list
# - toSave (str): Filename for saving the animation.
#
# Returns:
# None: This function does not return any value.
# It generates a 3D scatter plot animation and saves it as an MP4 video.
def ACP3Danimation(samplesInPCAspace, compToView, sampColorList, cluster_colors, samples, annotations, toSave):
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(samplesInPCAspace[:, compToView[0]],
                         samplesInPCAspace[:, compToView[1]],
                         samplesInPCAspace[:, compToView[2]],
                         zdir='z',
                         s=15,
                         c=sampColorList)

    custom_lines = [matplotlib.pyplot.Line2D([], [], ls="", marker='.', mec='k', mfc=c,
                                             mew=.1, ms=20) for c in cluster_colors.values()]

    legend_labels = [lt for lt in cluster_colors.keys()]

    # Divide the legends and lines into parts
    num_parts = 4
    part_size = len(custom_lines) // num_parts
    legend_parts = [custom_lines[i:i + part_size] for i in range(0, len(custom_lines), part_size)]
    label_parts = [legend_labels[i:i + part_size] for i in range(0, len(legend_labels), part_size)]

    # Handle the case where the last part is odd-sized
    if len(legend_parts[-1]) < part_size:
        last_part_lines = legend_parts.pop()
        last_part_labels = label_parts.pop()
        prev_part_lines = legend_parts[-1]
        prev_part_labels = label_parts[-1]
        # Add the remaining items to the previous part
        prev_part_lines.extend(last_part_lines)
        prev_part_labels.extend(last_part_labels)

    # Create subplots for the legend parts
    for i, (lines, labels) in enumerate(zip(legend_parts, label_parts)):
        ax_sub = fig.add_subplot(1, num_parts, i + 1)
        ax_sub.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1))
        ax_sub.axis('off')

    ax.set_xlabel("component " + str(compToView[0]))
    ax.set_ylabel("component " + str(compToView[1]))
    ax.set_zlabel("component " + str(compToView[2]))

    if len(annotations) > 0:
        for annot in annotations:
            ax.text3D(samplesInPCAspace[samples.index(annot), compToView[0]],
                      samplesInPCAspace[samples.index(annot), compToView[1]],
                      samplesInPCAspace[samples.index(annot), compToView[2]],
                      annot,
                      zdir='z')

    ani = matplotlib.animation.FuncAnimation(fig, animate, fargs=(ax, scatter), frames=range(0, 360, 10),
                                             interval=100, blit=True)

    # ani.save(toSave + '_animation.gif', writer='pillow')
    ani.save(toSave + '_animation.mp4', writer='ffmpeg')


def animate(angle, ax, scatter):
    ax.view_init(20, angle)
    return scatter,


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
