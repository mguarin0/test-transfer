# Interaction_Predictions

## Overview
This repository contains tools/instructions for calculating protein and molecular features as well as performing necessary preprocessing to produce a unit of analysis for ingestion by machine learning algorithms.

## Files in this repo
* [lib/cpi_preprocessing_environment.yml](lib/cpi_preprocessing_environment.yml) anaconda environment used for all preprocessing work
* [aws_build.sh](aws_build.sh) Script to download required packages on linux aws server.
* [msms](msms) Open source tool used to calculate residue depth.
* [stride](stride) Open source tool used to calculate relative accessible surface area.
* [test_input](test_input) Example pdb entry files used to test feature calculation scripts.
* [src](src) Location of feature calculation scripts.
* [src/unpackPDBB.py](src/unpackPDBB.py) This script unpacks the PDBBind dataset into separate protein, ligand, and pocket directories. All scripts below expect that this step has been performed.
* [src/runAll.py](src/runAll.py) Run all script for specified feature.
* [src/edgeFeatures.py](src/edgeFeatures.py) Edge feature calculation for protein.
* [src/hsaacFeatures.py](src/hsaacFeatures.py) Half sphere amino acid composition calculation for protein.
* [src/psaiaPDBBFormatter.py](src/psaiaPDBBFormatter.py) Format pdbb files to used on windows machine psaia tool.
* [src/psaiaFeatures.py](src/psaiaFeatures.py) Protrusion index and hydrophobicity calculation for protein.
* [src/residueDepthFeatures.py](src/residueDepthFeatures.py) Residue depth calculation for protein using msms tool.
* [src/strideFeatures.py](src/strideFeatures.py) Relative accessible surface area calculation for protein using stride tool.
* [src/psaiaPDBBFormatter.py](src/psaiaPDBBFormatter.py) Run this to format pdbb files to be used with psaia tool on windows.
* [src/pssmFeatures.py](src/pssmFeatures.py) Runs in two parts and performs Windowed Position Specific Scoring Matrix calculation.
* [src/moleculeFeatures.py](src/moleculeFeatures.py) Molecule features combines the vertex features (on atoms) and the edge features (on bonds).
* [src/interactionLabels.py](src/interactionLabels.py) Computes the interaction label between an atom from the molecule (ligand) and the atoms of a residue of a protein (receptor).
* [src/utils.py](src/utils.py) Contains utils for preprocessing.
* [src/foutDatasetExploration.ipynb](src/foutDatasetExploration.ipynb) Contains initial exploration of fout's dataset. This is kept for reference.
* [src/createComplexCodeDirs.py](src/createComplexCodeDirs.py) Organizes calculation files by complex code.
* [src/mergeFeatures.py](src/mergeFeatures.py) Merges all calculations into one master record pickle file to ingested by machine learning algorithm.

## To Run Feature Calculations
In order to run the feature calculations below you must download the [PDBBind dataset](http://www.pdbbind.org.cn/download.asp) and unpack it with `unpackPDBB.py` script using this command:
`python unpackPDBB.py <relative/full path to directory containing input data> <relative/full path to directory to unpack files to>`

### relative accessible surface area calculation
`python runAll.py r_stride <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### residue depth calculation calculation
`python runAll.py r_residueDepth <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### protrusion index and hydrophobicity calculation
`python runAll.py r_protrusion_idx <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### edge feature calculation
`python runAll.py r_edge <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### windowed position specific scoring matrix calculation (pssm)
This calculation uses the **psi-blast** tool to perform pssm calculation and requires that you have created a BLAST database from a [UniRef90](https://www.uniprot.org/downloads) fasta files. To create the BLAST database use the **makeblastdb** tool with this command from the directory that contains the BLAST executables: 

`makeblastdb -in uniref90.fasta -dbtype prot -out out_2` organize the output of this into a directory named `output_2`.

Now that you have created a BLAST database you can run the first step of the pssm calculation with this command:

`python runAll.py r_pssm_step_one <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin/` this script will output a fasta file from a PDBBind file, a pickle file containing all metadata info needed for joining record together during merge step, and performs pssm calculation using psi-blast and writes the results out. Note this calculation can take a very long time to run.

`python runAll.py r_pssm_step_two <relative/full path to directory output/r_pssm> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin/` this will output a pickle file to be merged into all protein features at a later step.

### protrusion index and hydrophobicity
`python psaiaPDBBFormatter.py <relative/full path to directory containing input data> <relative/full path to directory to output files to> <s3 bucket to move to>` use this script to format data to be used for windows psaia tool

Run psaia tool on windows machine, which will output to `.tbl` files containing protrusion index and hydrophobicity calculations. These results can be formatted with this command:

`python runAll.py r_protrusion_idx <relative/full path to directory output of psaia tool> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin/`

### half sphere amino acid composition calculation
`python runAll.py r_hsaac <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### molecule features
`python runAll.py molecule_features <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### interaction labels
`python runAll.py interaction_labels <relative/full path to directory containing input data> ../stride/stride ../msms/msms.x86_64Linux2.2.6.1.staticgcc ../ncbi-blast-2.7.1+/bin`

### merge all calculated features by complex code
`python createComplexCodeDirs.py <relative/full path to directory containing calculated features> <full path to merge directory>` this is the first part that must be run

`python mergeFeatures.py <full path to merge directory>` this merges all calculations performed on a complex into a record to be ingested by a machine learning algorithm

## aws set up notes

Launch instance instructions found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance)

Connect to instance instructions found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

To automate configuration of your EC2 instance run:`bash -i aws_build` and input your aws s3 credentials

## Open Source Tools Used to assist feature calculations

### PSI-BLAST tool
* [UniRef90](https://www.uniprot.org/downloads) fasta files needed for make BLAST database
* [BLAST cli tool](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download) BLAST executable tools.

### Stride tool
* [stride tool](http://ftp.ebi.ac.uk/pub/software/unix/stride/)

### Stride tool
* [msms tool](http://mgltools.scripps.edu/downloads#msms)

### PSAIA tool for windows:
* [psaia tool](http://bioinfo.zesoi.fer.hr/index.php/en/10-category-en-gb/tools-en/19-psaia-en)
* [get python](https://www.python.org/downloads/windows/)
* [aws-shell tool](https://github.com/awslabs/aws-shell)
* [aws shell commands](https://aws.amazon.com/cli/)

## Datasets
### PDBBind
* pdbb general download page: `http://www.pdbbind.org.cn/download.asp`
* Protein-ligand complexes: The general set minus refined set 11663 records `http://www.pdbbind.org.cn/download/pdbbind_v2018_other_PL.tar.gz`
* Protein-ligand complexes: The refined set 4463 records `http://www.pdbbind.org.cn/download/pdbbind_v2018_refined.tar.gz`
