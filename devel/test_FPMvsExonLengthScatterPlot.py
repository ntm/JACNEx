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
import numpy
import scipy.stats

####### JACNEx modules
import countFrags.countsFile
import countFrags.countFragments


logger = logging.getLogger(__name__)

# 23/02/2023
# NTM
#
# We currently normalize the counts as FPM, but we don't take into account
# the size of the exons.
# Question: shouldn't we normalize as FPMK = fragments per Million per KB of exon?
# See: https://github.com/septiera/JACNEx/issues/5#issuecomment-1434952464
#
# To answer this, we search here for a correlation between exon length and FPM.
#
# Conclusion: nothing stands out on the scatter plot, and the R2 is small
# -> no correlation, there is no point in normalizing in FPMK


countsFile = "../RunMageCNV/mageCNV_transcripts-230209/CountFiles/countsFile_2023-02-10_15-35-39.tsv.gz"


(exons, SOIs, countsArray) = countFrags.countsFile.parseCountsFile(countsFile)
countsFPM = countFrags.countFragments.normalizeCounts(countsArray)

exonLengths = numpy.array([exons[i][2] - exons[i][1] for i in range(len(exons))])

# average FPM per exon, over all samples
meanFPM = numpy.mean(countsFPM, axis=1)

#####################
# hum strange, some exons are huge!
# find their indexes in exons
numpy.where(exonLengths > 40000)[0]
# -> array([161869, 189096, 287231])
# retrieve their IDs:
exons[161869]
# -> ['chr11', 2608318, 2700004, 'ENST00000597346_1']
exons[189096]
# -> ['chr12', 102197575, 102402606, 'ENST00000626826_1']
exons[287231]
# -> ['chr22', 29978940, 30028246, 'ENST00000624945_1']
#
# checking in ensembl: AOK, these are all indeed huge lncRNAs
# back to my scatter plot
#####################

# enable interactive mode
# matplotlib.pyplot.ion()
fig, ax = matplotlib.pyplot.subplots()

# focus on exons that are captured (meanFPM >= 1) and not monstruously long (length<=2000)

# scatter plot
ax.scatter(exonLengths[(exonLengths < 500) & (meanFPM >= 1)], meanFPM[(exonLengths < 500) & (meanFPM >= 1)],
           marker=".", alpha=0.02)
matplotlib.pyplot.xlabel("exon length")
matplotlib.pyplot.ylabel("mean FPM (over all samples)")

matplotlib.pyplot.show()

# -> nothing obvious...

scipy.stats.linregress(exonLengths[(exonLengths < 2000) & (meanFPM >= 1)], meanFPM[(exonLengths < 2000) & (meanFPM >= 1)])
# -> LinregressResult(slope=0.0050747514980581864, intercept=2.8901257420726543, rvalue=0.5022837633435212, pvalue=0.0, stderr=1.9107703111450737e-05, intercept_stderr=0.007130762207770785)
# That's R-square = 0.25 ...

# Try with more focused exon length:
scipy.stats.linregress(exonLengths[(exonLengths < 500) & (meanFPM >= 1)], meanFPM[(exonLengths < 500) & (meanFPM >= 1)])
# -> LinregressResult(slope=0.008232802931077888, intercept=2.3805880584140873, rvalue=0.3701366521930766, pvalue=0.0, stderr=4.7446856525447423e-05, intercept_stderr=0.008476342902683063)
# even lower...

# Conclusion: no point in normalizing in FPMK
