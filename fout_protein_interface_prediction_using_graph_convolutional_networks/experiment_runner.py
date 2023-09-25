import os
import sys
import yaml
import traceback
import cPickle
import tensorflow as tf
import numpy as np
from configuration import data_directory, experiment_directory, output_directory, seeds, printt
from train_test import TrainTest
from pw_classifier import PWClassifier
from results_processor import ResultsProcessor
import logging

logging.basicConfig(filename='experiment_runner.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


# Load experiment specified in system args
exp_file = sys.argv[1]
printt("Running Experiment File: {}".format(exp_file))
f_name = exp_file.split(".")[0] if "." in exp_file else exp_file
exp_specs = yaml.load(open(os.path.join(experiment_directory, exp_file), 'r').read())

# setup output directory
outdir = os.path.join(output_directory, f_name)
if not os.path.exists(outdir):
    os.mkdir(outdir)
results_processor = ResultsProcessor()

# create results log
results_log = os.path.join(outdir, "results.csv")
with open(results_log, 'w') as f:
    f.write("")

# write experiment specifications to file
with open(os.path.join(outdir, "experiment.yml"), 'w') as f:
    f.write("{}\n".format(yaml.dump(exp_specs)))

# perform each experiment
prev_train_data_file = ""
prev_test_data_file = ""
first_experiment = True

# run through all experiments in yaml file
for experiment_name, experiment in exp_specs["experiments"]:
    train_data_file = os.path.join(data_directory, experiment["train_data_file"])
    test_data_file = os.path.join(data_directory, experiment["test_data_file"])
    try:
        # Reuse train data if possible.
        if train_data_file != prev_train_data_file:
            printt("Loading train data")
            logging.debug("Loading train data")
            train_list, train_data = cPickle.load(open(train_data_file))
            prev_train_data_file = train_data_file
        if test_data_file != prev_test_data_file:
            printt("Loading test data")
            logging.debug("Loading test data")
            test_list, test_data = cPickle.load(open(test_data_file))
            prev_test_data_file = test_data_file
        # create data dictionary
        data = {"train": train_data, "test": test_data}
        # perform experiment for each random seed
        for replica_number, seed_pair in enumerate(seeds):
            printt("{}: rep{}".format(experiment_name, replica_number))
            logging.debug("experiment_name: {} replica_number: {}".format(experiment_name, replica_number))
            # set tensorflow and numpy seeds
            tf.set_random_seed(seed_pair["tf_seed"])
            np.random.seed(int(seed_pair["np_seed"]))
            printt("Building model")
            # build model
            model = PWClassifier(experiment["layers"],
                                 experiment["layer_args"],
                                 data["train"],
                                 0.1,
                                 0.1,
                                 outdir,
                                 replica_number,
                                "train_{}".format(experiment_name))
            # train and test the model
            headers, results = TrainTest(results_processor=results_processor, replica_number=replica_number, experiment_name=experiment_name, outdir=outdir).fit_model(exp_specs, data, model)
            logging.debug("headers: {}".format(headers))
            logging.debug("results: {}".format(results))
            # write headers to file if haven't already
            if first_experiment:
                with open(results_log, 'a') as f:
                    f.write("{}\n".format(",".join(["file", "experiment", "rep", "specifications"] + headers)))
                first_experiment = False
            # write results to file
            with open(results_log, 'a') as f:
                f.write("{}, {}, {}, {}, {}\n".format(f_name, experiment_name, replica_number, format(experiment).replace(",", ";"), ",".join([str(r) for r in results])))
    except Exception as er:
        if er is KeyboardInterrupt:
            raise er
        ex_str = traceback.format_exc()
        logging.error(ex_str)
        printt(ex_str)
        printt("Experiment failed: {}".format(exp_specs))
