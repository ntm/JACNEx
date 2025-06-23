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
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_pdf
import scipy.cluster.hierarchy

# prevent matplotlib and PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


#############################
# visualize clustering results with a dendrogram.
# Most of the work is defining the leaf labels, which appear below the dendrogram,
# and the link colors, which should represent the constructed clusters.
# Args: see calling code in buildClusters()
# Produces plotFile (pdf format), returns nothing
def plotDendrogram(linkageMatrix, samples, clust2samps, fitWith, clustIsValid, title, plotFile):
    numSamples = len(samples)
    # build samp2clust for efficiency: key==sampleID, value==clusterID
    samp2clust = {}
    for clust in clust2samps.keys():
        for samp in clust2samps[clust]:
            samp2clust[samp] = clust

    ##################
    # leaf labels: we want to label the "middle" sample (visually in the dendrogram) of each
    # cluster with the clusterID eg 'A_02', other samples don't get labeled;
    # furthermore: if the cluster needs friends for fitting it gets lowercased eg 'a_02', and if it
    # is invalid it is lowercased and parenthesized eg '(a_02)'
    # key == sample index in samples (only defined for samples that get labeled), value == label
    sampi2label = {}
    # Also populate clust2color: key==clusterID, value == matplotlib color string code, we want to
    # cycle over a palette in the visual dendrogram order
    clust2color = {}
    allColors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    # next color index (in allColors)
    nextColi = 0

    # produce a first virtual dendrogram to get the order of samples/leaves
    R = scipy.cluster.hierarchy.dendrogram(linkageMatrix, get_leaves=True, count_sort='descending', no_plot=True)
    pos2si = R['leaves']
    # pos2si[pos] == si  <=> the leave at position pos (left-to-right, < numSamples) is sample index si

    nextPos = 0
    while nextPos < numSamples:
        nextSample = samples[pos2si[nextPos]]
        nextClust = samp2clust[nextSample]
        # color for nextClust
        clust2color[nextClust] = allColors[nextColi]
        nextColi += 1
        if nextColi == len(allColors):
            nextColi = 0
        # leaves from nextPos to nextPos+len(clust2samps[nextClust])-1 belong to cluster nextClust, label
        # the leaf in the middle
        posToLabel = nextPos + (len(clust2samps[nextClust]) - 1) // 2

        # create label for this cluster, "decorated" if it needs friends / is invalid
        label = ""
        if not clustIsValid[nextClust]:
            label = '(' + nextClust.lower() + ')'
        elif len(fitWith[nextClust]) > 0:
            label = nextClust.lower()
        else:
            label = nextClust

        sampi2label[pos2si[posToLabel]] = label
        nextPos += len(clust2samps[nextClust])

    # finally we can define the leaf label function
    def leafLabelFunc(sampi):
        if sampi in sampi2label:
            return sampi2label[sampi]
        else:
            return('')

    ##################
    # log sampleIDs + clusterIDs from left to right in the dendrogram
    legendFile = plotFile.removesuffix('.pdf') + '.txt'
    try:
        legendFH = open(legendFile, 'w')
    except Exception as e:
        logger.error("Cannot open dendrogram legend file %s: %s", legendFile, e)
        raise Exception('cannot open dendrogram legend file for writing')

    legendFH.write("dendrogram leaves from left to right:\n")
    for pos in range(numSamples):
        sample = samples[pos2si[pos]]
        clust = samp2clust[sample]
        legendFH.write(clust + " - " + sample + "\n")
    legendFH.close()

    ##################
    # link colors: one color for each clusterID
    # for each non-singleton node i (== row index in linkageMatrix), populate node2clust:
    # node2clust[i] == the clusterID that node i belongs to if it belongs to a cluster,
    # or "V:clusterID" if node i is a "virtual merger" of several (unmerged) clusters including
    # clusterID (the choice of the clusterID to use is arbitrary and doesn't matter)
    node2clust = [None] * linkageMatrix.shape[0]

    # set n2c[c] = clust for non-singleton nodei and all its non-singleton descendants
    # whose pre-existing n2c value starts with 'V:'
    def setVirtualsToClust(n2c, nodei, clust):
        if n2c[nodei].startswith('V:'):
            n2c[nodei] = clust
            for ci in range(2):
                child = int(linkageMatrix[nodei, ci])
                if child >= numSamples:
                    setVirtualsToClust(n2c, child - numSamples, clust)

    for node in range(linkageMatrix.shape[0]):
        (c1, c2, dist, size) = linkageMatrix[node]
        c1 = int(c1)
        c2 = int(c2)
        if c1 < numSamples:
            clust1 = samp2clust[samples[c1]]
        else:
            clust1 = node2clust[c1 - numSamples]
        if c2 < numSamples:
            clust2 = samp2clust[samples[c2]]
        else:
            clust2 = node2clust[c2 - numSamples]

        if clust1.startswith('V:') and clust2.startswith('V:'):
            # both are virtual mergers, arbitrarily use the first
            node2clust[node] = clust1
        elif clust1.startswith('V:') or clust2.startswith('V:'):
            # switch if needed so c1 is the virtual
            if clust2.startswith('V:'):
                ctmp = c2
                c2 = c1
                c1 = ctmp
                ctmp = clust2
                clust2 = clust1
                clust1 = ctmp
            virtClust = clust1.removeprefix('V:')
            if virtClust in fitWith[clust2]:
                # node and all its 'V:' descendants are "given" to clust2 (for coloring)
                node2clust[node] = clust2
                setVirtualsToClust(node2clust, c1 - numSamples, clust2)
            else:
                # define node as Virtual
                node2clust[node] = clust1
        else:
            # c1 and c2 are both non-virtual
            if (clust1 == clust2):
                node2clust[node] = clust1
            elif (clust2 in fitWith[clust1]):
                node2clust[node] = clust1
            elif (clust1 in fitWith[clust2]):
                node2clust[node] = clust2
            else:
                node2clust[node] = 'V:' + clust1

    # finally we can define the node/link color function
    def linkColorFunc(node):
        node -= numSamples
        if node2clust[node].startswith('V:'):
            # node doesn't belong to a non-virtual cluster, default to black
            return('k')
        else:
            return(clust2color[node2clust[node]])

    ##################
    # make the plot
    matplotlib.use('pdf')
    pdf = matplotlib.backends.backend_pdf.PdfPages(plotFile)
    fig = matplotlib.figure.Figure(figsize=(25, 10))
    ax = fig.subplots()
    scipy.cluster.hierarchy.dendrogram(linkageMatrix,
                                       leaf_rotation=30,
                                       leaf_label_func=leafLabelFunc,
                                       link_color_func=linkColorFunc,
                                       count_sort='descending',
                                       ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    pdf.savefig(fig)
    pdf.close()
