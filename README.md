<h1 align="center"> JACNEx: Joint Analysis of Copy Numbers from Exomes</h1>

JACNEx calls germline Copy Number Variations (CNVs) from exome sequencing data.
It is sensitive, specific, and very fast.

JACNEx performs the following steps:
1. **Count fragments**: counts the number of sequenced fragments from each BAM file that overlap each exon.
2. **Cluster samples**: identify clusters of "similar" samples (typically captured with the same kit, usually
in the same center, and other factors can come into play). Clusters are built independantly for autosomes and
for gonosomes.
3. **Call CNVs**:
   - fit a distribution to model CN0 using intergenic pseudo-exons;
   - fit a distribution to model CN2 for each exon in each cluster;
   - identify non-interpretable exons in each cluster (extensive quality control);
   - calculate the likelihood of each CN state (CN0, CN1, CN2, CN3+) for each interpretable exon in each cluster;
   - fit prior probabilities and distance-dependant transition probabilities for each cluster;
   - combine all of this to construct a Continuous Hidden Markov Model for each cluster, and use it to call CNVs.


JACNEx is fast in general, but it was particularly engineered for speed in the incremental use case, i.e.:
you have a (large) collection of exomes, and you regularly add some new exomes to this collection.
The new exomes can help improve the CNV calls of some of the previous samples, but some tasks
that were done previously should not need to be repeated (eg step 1, count fragments).
JACNEx solves this correctly by saving key intermediate data each time you run it.
Then when you run it again with some additional samples, it re-uses the previously saved data when
possible, while redoing everything that could have different results.
It is therefore extremely fast in this (frequent) incremental N+1 use case, without any compromise on the results.


**JACNEx input**:
- BAM files, one per sample;
- a BED file defining the exons to consider.


**JACNEx output**:
- CNVs in VCF format are saved as VCFs/CNVs_[date].vcf.gz .
- The clusters from step 2 are also saved as interpretable TSV files in the Clusters/ subdir.
  These can be very useful for quality control, eg since gonosome clusters are always single-gender
  they allow to identify mis-labeled samples (wrong gender).
- If called with `--plotCNVs`, JACNEx produces PDF files in the Plots_CNVs/ subdir for each sample,
  containing plots for each exon in and immediately preceding/following each called CNV.
  Producing these plots slows JACNEx down somewhat, but we find them very useful for QC and validation,
  particularly for CNVs with lower GQ scores.


### How to use JACNEx
1. [Install JACNEx and its dependencies](#installing)
2. [Prepare input files](#preparing-input)
3. [Run JACNEx](#running-jacnex)


<hr>

### Installing
We try to keep dependencies to a minimum, but we still have a few: we rely on samtools,
as well as a small number of python modules. Everything can be cleanly installed as follows.

#### JACNEx
`git clone https://github.com/ntm/JACNEx.git`

#### samtools
JACNEx needs samtools (tested with v1.15.1 - v1.21), it can be installed with: <br>
```
VER=1.21
wget https://github.com/samtools/samtools/releases/download/$VER/samtools-$VER.tar.bz2

tar xfvj samtools-$VER.tar.bz2
cd samtools-$VER
./configure
make all
```
You then need to place samtools-$VER/samtools in your $PATH (e.g. create a symlink to it in
/usr/local/bin/ if you are sudoer), or pass it to JACNEx.py with `--samtools=` .

#### Python
JACNEx needs python version >= 3.7 (3.6 and earlier have a bug that breaks JACNEx).
For example on ALMA Linux 9 we use python3.12, available in the standard repos since ALMA 9.4:
```
sudo dnf install python3.12 python3.12-setuptools python3.12-numpy python3.12-scipy
sudo dnf install python3.12-pip-wheel python3.12-setuptools-wheel python3.12-wheel-wheel
```

#### Python modules
JACNEx requires the following python modules:<br>
**numpy scipy numba ncls matplotlib scikit-learn KDEpy**<br>
On some distributions/environments you may also need to (re-)install setuptools.
We recommend the following commands, which cleanly install all the requirements in
a python virtual environment (in ~/pyEnv_JACNEx/), using the system-wide versions if available:
```
PYTHON=python3.12 ### or python3, or python, or...
$PYTHON -m venv --system-site-packages ~/pyEnv_JACNEx
source ~/pyEnv_JACNEx/bin/activate
pip install --upgrade pip
pip install setuptools numpy scipy numba ncls matplotlib scikit-learn KDEpy
```
On an ALMA9.4 system today (28/05/2024) with python3.12 installed as suggested above, this uses
the system-wide:<br>
setuptools-68.2.2 numpy-1.24.4 scipy-1.11.1

and it installs in ~/pyEnv_JACNEx/ :<br>
numba-0.59.1 ncls-0.0.68 matplotlib-3.8.4 scikit_learn-1.4.2 KDEpy-1.1.9

You then need to activate the venv before running JACNEx, e.g.:
```
$ source ~/pyEnv_JACNEx/bin/activate
(pyEnv_JACNEx) $ python path/to/JACNEx/JACNEx.py --help
```

<hr>

### Preparing input
JACNEx needs a **BED file of exons**. Sequenced fragments that overlap each provided exon
will be counted, and called CNVs will be comprised of one or more consecutive exons.
JACNEx applies several quality controls throughout, including filtering of exons that
are not captured or not correctly covered by unambiguous alignments. Therefore the BED
can be extensive, and doesn't have to be restricted to exons that are targeted by your
capture kit.
For example, for Human exomes we currently use exons of the Ensembl canonical transcripts,
specifically the canonical_$VERS.bed.gz file produced as
[documented here](https://github.com/ntm/grexome-TIMC-Secondary/tree/master/Transcripts_Data).

JACNEx also needs **BAM files**, one file for each sample. Each BAM filename (stripped of .bam)
will be used to identify the corresponding sample, i.e. each file should be named `<sampleID>.bam` .
The filenames can be provided on the command line with `--bams=` , or they can be listed in a text file,
one filename per line, with `--bams-from=` .


<hr>

### Running JACNEx
This is (almost) how we currently (24/10/2025) run JACNEx in production each time we sequence new exomes,
on our regularly-growing collection of ~1300 human exomes with new samples being added 1-4 times per month.
```
source ~/pyEnv_JACNEx/bin/activate
TRANS=canonical_115
python3 ~/Software/JACNEx/JACNEx.py --bams-from allBams_251022.list --bed ~/Transcripts_Data/${TRANS}.bed.gz --workDir JACNEx_${TRANS} --plotCNVs --tmp /mnt/RamDisk/ --jobs 60 2> JACNEx.${TRANS}.`date +%y%m%d`.log &
```

You can see all possible command-line arguments with:
```
source ~/pyEnv_JACNEx/bin/activate
python3 ~/Software/JACNEx/JACNEx.py --help
```
