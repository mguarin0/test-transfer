import numpy as np
np.set_printoptions(threshold=np.nan)

from configuration import printt
from tqdm import tqdm
import logging

class TrainTest:
    def __init__(self, replica_number, experiment_name, outdir, results_processor=None):
        self.results_processor = results_processor
        self.replica_number = replica_number
        self.experiment_name = experiment_name
        self.outdir = outdir

    def fit_model(self, exp_specs, data, model):
        """
        trains model by iterating minibatches for specified number of epochs
        """
        printt("Fitting Model")
        # train for specified number of epochs
        for epoch in range(1, exp_specs["num_epochs"] + 1):
            logging.debug("===========TRAINING LOOP NEXT EPOCH==========")
            self.train_epoch(data["train"], model, exp_specs["minibatch_size"], (epoch, exp_specs["num_epochs"]))
        # calculate train and test metrics
        headers, result = self.results_processor.process_results(exp_specs, data, model, "{}_{}".format(self.experiment_name, self.replica_number), self.outdir)
        logging.debug("|| TrainTest.fit_model ||  headers: {}".format(headers))
        logging.debug("|| TrainTest.fit_model ||  result: {}".format(result))
        # clean up
        self.results_processor.reset()
        model.close()
        return headers, result

    def train_epoch(self, data, model, minibatch_size, epoch_info):
        """
        Trains model for one pass through training data, one complex at a time
        Each complex is split into minibatches of paired examples.
        Features for the entire complex are passed to model,
        but only a minibatch of examples are passed
        """
        complex_permutation = np.random.permutation(len(data))

        # loop through each complex
        for complex in tqdm(complex_permutation, "|| training epoch {} of {} || deploying batches".format(*epoch_info)):
            # extract just data for this complex
            complex_data = data[complex]
            pair_examples = complex_data["label"]
            n = len(pair_examples)
            # random vertices indices
            shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
            # loop through each minibatch to complete training epoch
            for i in range(int(n / minibatch_size)):
                # extract data for this minibatch
                index = int(i * minibatch_size)
                # grab examples to use
                examples = pair_examples[shuffle_indices[index: index + minibatch_size]]
                minibatch = {}
                for feature_type in complex_data:
                    if feature_type == "label":
                        minibatch["label"] = examples
                    else:
                        minibatch[feature_type] = complex_data[feature_type]
                # train the model
                model.train(minibatch)
