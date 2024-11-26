<h1 align="center"> Call CNVs from exome sequencing data </h1>

JACNEx calls germline Copy Number Variations (CNVs) from exome sequencing data.<br>
The input data of this pipeline are Binary Alignment Maps (BAM) and Browser Extensible Data (BED) containing the intervals associated with the canonical transcripts.<br>
For more information how obtaining the different files see https://github.com/ntm/grexome-TIMC-Primary<br>

##### STEP 1 : Prepare the Environment<br>

Before running JACNEx, ensure all dependencies are installed, and the environment is set up correctly.<br>

##### STEP 2 : Run JACNEx<br>

JACNEx.py orchestrates the entire pipeline, running the necessary steps to count fragments, cluster samples, and call CNVs. This involves the following steps:<br>

1. **Fragments Counting**: Counts the number of sequenced fragments from each BAM file that overlap each exon.<br>
2. **Samples Clustering**: Normalizes the counts, performs quality control on the samples, and forms the reference clusters.<br>
3. **Copy Number Calling**: Determines parameters for CN0 and CN2 distributions, excludes non-interpretable exons, calculates likelihoods for each CN state, and applies a Continuous Hidden Markov Model (CO-HMM) to call and group CNVs.<br>

The pipeline also handles intermediate results, quality control checks, and outputs the CNV calls in VCF format.<br>

To run JACNEx.py, use the following command syntax:
```
BAMS="BAMs/sample1.bam,BAMs/sample2.bam"
BED="Transcripts_Data/ensemblCanonicalTranscripts.bed.gz"
WORKDIR="/path/to/workDir"

python JACNEx.py --bams $BAMS --bed $BED --workDir $WORKDIR 2> jacnex.log
```

### CONFIGURATION:


### DEPENDENCIES:

#### samtools
JACNEx needs samtools (tested with v1.15.1 - v1.18), can be installed with: <br>
```
wget https://github.com/samtools/samtools/releases/download/1.18/samtools-1.18.tar.bz2
tar xfvj samtools-1.18.tar.bz2
cd samtools-1.18
./configure
make all all-htslib
```
You then need to place samtools-1.18/samtools in your $PATH (e.g. create a symlink to it in /usr/local/bin/ if you are sudoer), or pass it to JACNEx.py with --samtools= .

#### python
JACNEx needs python version >= 3.7 (3.6 and earlier have a bug that breaks JACNEx).
For example on ALMA Linux 9 we use python3.12, available in the standard repos since ALMA 9.4:
```
sudo dnf install python3.12 python3.12-setuptools python3.12-numpy python3.12-scipy
sudo dnf install python3.12-pip-wheel python3.12-setuptools-wheel python3.12-wheel-wheel
```

#### python modules
JACNEx requires the following python modules:<br>
_numpy scipy numba ncls matplotlib pyerf scikit-learn KDEpy_<br>
On some distributions/environments you may also need to (re-)install setuptools.
We recommend the following commands, which cleanly install all the requirements in
a python virtual environment, using the system-wide versions if available:
```
PYTHON=python3.12 ### or python3, or python, or...
$PYTHON -m venv --system-site-packages ~/pyEnv_JACNEx
source ~/pyEnv_JACNEx/bin/activate
pip install --upgrade pip
pip install setuptools numpy scipy numba ncls matplotlib pyerf scikit-learn KDEpy
```
On an ALMA9.4 system today (28/05/2024) this uses the system-wide:<br>
**setuptools-68.2.2 numpy-1.24.4 scipy-1.11.1**

(provided they are installed on the system), and it installs in ~/pyEnv_JACNEx/ :<br>
**numba-0.59.1 ncls-0.0.68 matplotlib-3.8.4 pyerf-1.0.1 scikit_learn-1.4.2 KDEpy-1.1.9**

You then need to activate the venv before running JACNEx, e.g.:
```
$ source ~/pyEnv_JACNEx/bin/activate
(pyEnv_JACNEx) $ python path/to/JACNEx/JACNEx.py --help
```
