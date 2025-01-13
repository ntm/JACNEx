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
import matplotlib.pyplot
import matplotlib.backends.backend_pdf
import numpy
import os
import scipy.cluster.hierarchy

# prevent matplotlib and PIL flooding the logs when we are in DEBUG loglevel
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################

###################################
# Plot one or more curves, and optionally two vertical dashed lines, on a
# single figure.
# Each curve is passed as an ndarray of X coordinates (eg dataRanges[2] for the
# third curve), a corresponding ndarray of Y coordinates (densities[2]) of the
# same length, and a legend (legends[2]).
# The vertical dashed lines are drawn at X coordinates line1 and line2, unless
# line1==line2==0.
#
# Args:
# - title: plot's title (string)
# - dataRanges: list of N ndarrays storing X coordinates
# - densities: list of N ndarrays storing the corresponding Y coordinates
# - legends: list of N strings identifying each (dataRange,density) pair
# - line1, line2 (floats): X coordinates of dashed vertical lines to draw
# - line1legend, line2legend (strings): legends for the vertical lines
# - ylim (float): Y max plot limit
# - pdf: matplotlib PDF object where the plot will be saved
#
# Returns nothing.
def plotDensities(title, dataRanges, densities, legends, line1, line2, line1legend, line2legend, ylim, pdf):
    # sanity
    if (len(dataRanges) != len(densities)) or (len(dataRanges) != len(legends)):
        raise Exception('plotDensities bad args, length mismatch')

    # set X max plot limit (both axes start at 0)
    xlim = max(dataRanges[:][-1])

    # Disable interactive mode
    matplotlib.pyplot.ioff()
    fig = matplotlib.pyplot.figure(figsize=(6, 6))
    for i in range(len(dataRanges)):
        matplotlib.pyplot.plot(dataRanges[i], densities[i], label=legends[i])

    if (line1 != 0) or (line2 != 0):
        matplotlib.pyplot.axvline(line1, color='crimson', linestyle='dashdot', linewidth=1, label=line1legend)
        matplotlib.pyplot.axvline(line2, color='darkorange', linestyle='dashdot', linewidth=1, label=line2legend)

    matplotlib.pyplot.xlabel("FPM")
    matplotlib.pyplot.ylabel("density")
    matplotlib.pyplot.xlim(0, xlim)
    matplotlib.pyplot.ylim(0, ylim)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc='upper right', fontsize='small')

    pdf.savefig(fig)
    matplotlib.pyplot.close()


#############################################################
# plotPieChart: ### DEV deprecate script and replace with DEBUG message in s3_callCNVs.py
# Generate and save a pie chart representing the distribution of exon status
# (filtered or called).
#
# Args:
# - clustID [str]: cluster identifier
# - filterStates (list[strs]): filter states IDs
# - exStatusArray (numpy.ndarray[ints]): exon filtering states indexes
# - plotDir: Folder path for save the generated plot
def plotPieChart(clustID, filterStates, exStatusArray, plotDir):
    name_pie_charts_file = f"exonsFiltersSummary_pieChart_cluster_{clustID}.pdf"
    pdf_pie_charts = os.path.join(plotDir, name_pie_charts_file)
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf_pie_charts)

    # Use numpy.unique() to obtain unique values and their occurrences
    # with return_counts=True
    uniqueValues, counts = numpy.unique(exStatusArray[exStatusArray != -1], return_counts=True)

    # Create the pie chart figure and subplot
    fig = matplotlib.pyplot.figure(figsize=(5, 5))
    ax11 = fig.add_subplot(111)

    # Plot the pie chart with customization
    w, l, p = ax11.pie(counts,
                       labels=None,
                       autopct=lambda x: str(round(x, 2)) + '%',
                       textprops={'fontsize': 14},
                       startangle=160,
                       radius=0.5,
                       pctdistance=1,
                       labeldistance=None)

    # Calculate percentage distances for custom label positioning
    step = (0.8 - 0.2) / (len(filterStates) - 1)
    pctdists = [0.8 - i * step for i in range(len(filterStates))]

    # Position the labels at custom percentage distances
    for t, d in zip(p, pctdists):
        xi, yi = t.get_position()
        ri = numpy.sqrt(xi**2 + yi**2)
        phi = numpy.arctan2(yi, xi)
        x = d * ri * numpy.cos(phi)
        y = d * ri * numpy.sin(phi)
        t.set_position((x, y))

    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.title("Filtered and called exons from cluster " + str(clustID))
    matplotlib.pyplot.legend(loc='upper right', fontsize='small', labels=filterStates)
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    matplotOpenFile.close()


#############################################################
# barPlot ### DEV deprecate script
# Creates a bar plot of copy number frequencies based on the count array.
# The plot includes error bars representing the standard deviation.
#
# Args:
# - countArray (numpy.ndarray[ints]): Count array representing copy number frequencies.
# - CNStatus (list[str]): Names of copy number states.
# - outFolder (str): Path to the output folder for saving the bar plot.
def barPlot(countArray, CNStatus, pdf):
    matplotOpenFile = matplotlib.backends.backend_pdf.PdfPages(pdf)
    fig = matplotlib.pyplot.figure(figsize=(10, 8))

    # Calculate the mean and standard deviation for each category
    means = numpy.mean(countArray, axis=0)
    stds = numpy.std(countArray, axis=0)

    # Normalize the means to get frequencies
    total_mean = numpy.sum(means)
    frequencies = means / total_mean

    # Plot the bar plot with error bars
    matplotlib.pyplot.bar(CNStatus, frequencies, yerr=stds / total_mean, capsize=3)

    # Define the vertical offsets for the annotations dynamically based on standard deviation
    mean_offset = numpy.max(frequencies) * 0.1
    std_offset = numpy.max(frequencies) * 0.05

    # Add labels for mean and standard deviation above each bar
    for i in range(len(CNStatus)):
        matplotlib.pyplot.text(i, frequencies[i] + mean_offset, f'μ={frequencies[i]:.1e}', ha='center')
        matplotlib.pyplot.text(i, frequencies[i] + std_offset, f'σ={stds[i] / total_mean:.1e}', ha='center')

    # Set the labels and title
    matplotlib.pyplot.xlabel('Copy number States')
    matplotlib.pyplot.ylabel('Frequencies')
    matplotlib.pyplot.title(f'CN frequencies Bar Plot for {len(countArray)} samps (Excluding Filtered)')
    matplotOpenFile.savefig(fig)
    matplotlib.pyplot.close()
    matplotOpenFile.close()
