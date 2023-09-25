###  This small script runs the experiments from our paper.  ###
###  Uncomment a line to run the respective experiment.      ###
###  These experiments are actually run multiple times,      ###
###  once for each set of random seeds in configuration.py.  ###
###  For this reason we recommend not uncommenting more      ###
###  than one line at a time.                                ###

export PL_DATA=./pl_data
export PL_OUT=./pl_out
export PL_EXPERIMENTS=./pl_experiments
export CUDA_VISIBLE_DEVICES=0

#python experiment_runner.py no_conv.yml
#python experiment_runner.py node_average.yml
python experiment_runner.py node_edge_average.yml
