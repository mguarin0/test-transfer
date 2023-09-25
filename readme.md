# Interaction_Predictions

## Overview
This repo contains work/experiments for protein-molecule interaction prediction.

## Files in this repo
* Official implementation of ["Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences"](./tsubaki_compound_protein_interaction/)
* Official implementation of ["Protein Interface Prediction using Graph Convolutional Networks"](./fout_protein_interface_prediction_using_graph_convolutional_networks)
 
## Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequence 

### Experiment Details
<img src="assets/Compound-protein Interaction Prediction with End-to-end Learning of GNN and CNN architecture.png" align="center">
This work uses a graph convolutional neural network and a standard convolutional neural network to perform compound protein interaction prediction as binary classification. DUD-E CPI dataset used with 6728 examples.
Dataset used provides SMILES for molecules which are trasnformed to 2D molecular graphs using RDKit and proteins are represented as 1D sequential data where every element in the sequence is an amino acid. For every example a binary label is provided to indicate if interaction occurs.

### Results
* Results after after 99 training epochs on 6728 examples:
	* Loss=22.0821014
	* AUC_dev=0.967213
	* AUC_test=0.973574
	* Precision=0.956647
	* Recall=0.924581

* Results after after 99 training epochs on ~70,000 examples:
	* Loss=7346.433704294264
	* AUC_dev=0.9330775555381126
	* AUC_test=0.9410998218132827
	* Precision=0.7966101694915254
	* Recall=0.5172955974842768

### Claimed Results
<img src="assets/Compound-protein Interaction Prediction with End-to-end Learning of GNN and CNN architecture_figure5.png" align="center">

<img src="assets/Compound-protein Interaction Prediction with End-to-end Learning of GNN and CNN architecture_figure6.png" align="center">

<img src="assets/Compound-protein Interaction Prediction with End-to-end Learning of GNN and CNN architecture_figure7.png" align="center">

<img src="assets/Compound-protein Interaction Prediction with End-to-end Learning of GNN and CNN architecture_figure8.png" align="center">

### Future work
There are two options for improving the performance of this model for the task of compound protein interaction prediction:
1. gather more data and see if it improves performance
2. experiment with changes in the model architecture to improve performance

Currently (as of October 17th) the most efficient approach to improve model performance is simply to gather more data as the type of data (SMILES, sequence of amino acids, binary interaction label) is readily available rather than explore methods to improve this model's architecture. Once performance has been evaluated on a larger dataset we will reevaluate a plan of action.

## Protein Interface Prediction using Graph Convolutional Networks

### Experiment Details
<img src="assets/Protein Interface Prediction using Graph Convolutional Networks.png" align="center">
In this work proteins are represented as graphs where each amino acid residue is a node whose features represent the properties of the residue; the spatial relationships between residues (distances, angles) are represented as features of the edges that connect them. The neighborhood of a node used in the convolution operator is the set of k closest residues as determined by the mean distance between their atoms.

This work uses a Version 5 of the Docking Benchmark Dataset (DBD) which is the standard benchmark dataset for assessing docking and interface prediction methods and is a carefully selected subset of structures from the Protein Data Bank (PDB). These proteins range in length from 29 to 1979 residues with a median length of 203.5. 140 complexes are used during training, 35 complexes are used during validation, and 55 complexes were used during testing.

There are several different experiments to determine the optimal method for graph convolution {order dependent, dtnn, node and edge averaging, node averaging, single weight matrix, diffusion (2hops), diffusion (5hops), no convolution}.

### Results
currently running the _Order Dependent_ method (equation3) for graph convolutional which is claimed to be the best performing variant of this model for this task.

### Claimed Results
<img src="assets/Protein Interface Prediction using Graph Convolutional Networks_figure2.png" align="center">

### Future work
Current we have a meeting with Alex Fout this friday (October 19) to more deeply understand the motivations of his model and also the input data that he is using.

**Author's notes on future work:**

Notes on limitation of current models...

"Our experiments did not demonstrate a big difference with the inclusion of edge features. There were very few of those, and unlike the node features, they were static: our networks learned latent representations only for the node features. These methods can be extended to learn both node and edge representations, and the underlying convolution operator admits a simple deconvolution operator which lends itself to be used with auto-encoders."

Thoughts on a move towards a unsupervised approach in order to leverage more data...

"CNNs typically require large datasets to learn effective representations. This may have limited the level of accuracy that we could attain using our purely supervised approach and the relatively small
number of labeled training examples. Unsupervised pre-training would allow us to use the entire Protein Data Bank which contains close to 130,000 structures"

Thoughts on using structural data for proteins...

"The analogous level of description for protein structure would be the raw 3D atomic coordinates, which we thought would prove too difficult. Using much larger training sets and unsupervised learning can potentially allow the network to begin with features that are closer to the raw atomic coordinates and learn a more detailed representation of the geometry of proteins"
