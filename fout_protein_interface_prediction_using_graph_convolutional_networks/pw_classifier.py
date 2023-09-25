import os
import copy

import numpy as np
import tensorflow as tf

import nn_components
import logging

__all__ = [
    "PWClassifier",
]


class PWClassifier(object):
    def __init__(self, layer_specs, layer_args, train_data, learning_rate, pn_ratio, outdir, replica_number, experiment_name):
        """ Assumes same dims and nhoods for l_ and r_ """
        self.replica_number = replica_number
        self.experiment_name = experiment_name
        self.layer_args = layer_args
        self.params_r = {}
        self.params_l = {}
        self.params_md = {}
        # tf stuff:
        self.graph = tf.Graph()
        self.sess = None
        self.preds = None
        self.labels = None
        #################################################################
        # get details of data

        self.in_nv_dims_r = train_data[0]["r_vertex"].shape[-1] # vertex feature dim (70 for fout)
        self.in_nv_dims_l = train_data[0]["l_vertex"].shape[-1] # vertex feature dim (70 for fout)
        self.in_ne_dims_r = train_data[0]["r_edge"].shape[-1] # edge feature dimensions (2 for fout) 
        self.in_ne_dims_l = train_data[0]["l_edge"].shape[-1] # edge feature dimensions (2 for fout) 
        self.in_nhood_size_r = train_data[0]["r_hood_indices"].shape[1] # number of neighbors in neighborhood (20 for fout)
        self.in_nhood_size_l = train_data[0]["l_hood_indices"].shape[1] # number of neighbors in neighborhood (20 for fout)
        with self.graph.as_default():
            # define placeholders
            self.in_vertex_r = tf.placeholder(tf.float32, [None, self.in_nv_dims_r], "vertex_r")
            self.in_vertex_l = tf.placeholder(tf.float32, [None, self.in_nv_dims_l], "vertex_l")
            self.in_edge_r = tf.placeholder(tf.float32, [None, self.in_nhood_size_r, self.in_ne_dims_r], "edge_r")
            self.in_edge_l = tf.placeholder(tf.float32, [None, self.in_nhood_size_l, self.in_ne_dims_l], "edge_l")
            self.in_hood_indices_r = tf.placeholder(tf.int32, [None, self.in_nhood_size_r, 1], "hood_indices_r")
            self.in_hood_indices_l = tf.placeholder(tf.int32, [None, self.in_nhood_size_l, 1], "hood_indices_l")
            input_r = self.in_vertex_r, self.in_edge_r, self.in_hood_indices_r
            input_l = self.in_vertex_l, self.in_edge_l, self.in_hood_indices_l
            self.examples = tf.placeholder(tf.int32, [None, 2], "examples") # minibatch of examples passed to model
            self.labels = tf.placeholder(tf.float32, [None], "labels") # label for minibatch of examples
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")
            #### make layers
            legs = True
            i = 0
            # run through layer specifications
            while i < len(layer_specs):
                layer = layer_specs[i]
                args = copy.deepcopy(layer_args)
                args["dropout_keep_prob"] = self.dropout_keep_prob
                type = layer[0]
                args2 = layer[1] if len(layer) > 1 else {}
                flags = layer[2] if len(layer) > 2 else None
                args.update(args2)  # local layer args override global layer args
                layer_fn = getattr(nn_components, type)
                # if "merge" flag is in this layer, then this is a merge layer and every subsequent layer is a merged layer
                if flags is not None and "merge" in flags:
                    legs = False
                    input_md = input_r[0], input_l[0], self.examples  # take vertex features only
                if legs:
                    # make leg layers (everything up to the merge layer)
                    name = "leg_r_{}_{}".format(type, i)
                    with tf.name_scope(name):
                        output_r, params_r = layer_fn(input_r, None, leg_type="r", experiment_name=self.experiment_name, replica_number=self.replica_number, layer=i, **args)
                        if params_r is not None:
                            self.params_r.update({"{}_{}".format(name, k): v for k, v in params_r.items()})
                        input_r = output_r, self.in_edge_r, self.in_hood_indices_r
                    name = "leg_l_{}_{}".format(type, i)
                    with tf.name_scope(name):
                        output_l, params_l = layer_fn(input_l, None, leg_type="l", experiment_name=self.experiment_name, replica_number=self.replica_number, layer=i, **args)
                        if params_l is not None:
                            self.params_l.update({"{}_{}".format(name, k): v for k, v in params_l.items()})
                        input_l = output_l, self.in_edge_l, self.in_hood_indices_l
                else:
                    # merged layers
                    name = "{}_{}".format(type, i)
                    with tf.name_scope(name):
                        input_md, params_md = layer_fn(input_md, None, experiment_name=self.experiment_name, replica_number=self.replica_number, layer=i, **args)
                        if params_md is not None and len(params_md.items()) > 0:
                            self.params_md.update({"{}_{}".format(name, k): v for k, v in params_md.items()})
                i += 1
            self.preds = input_md

            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.inc = tf.assign_add(self.global_step, 1, name="incrememt") 

            # Loss and optimizer
            with tf.name_scope("loss"):
                scale_vector = (pn_ratio * (self.labels - 1) / -2) + ((self.labels + 1) / 2)
                logits = tf.concat([-self.preds, self.preds], axis=1)
                labels = tf.stack([(self.labels - 1) / -2, (self.labels + 1) / 2], axis=1)
                self.loss = tf.losses.softmax_cross_entropy(labels, logits, weights=scale_vector)
                tf.summary.scalar("{}_{}/scalar_loss".format(self.experiment_name, self.replica_number), self.loss)

            with tf.name_scope("optimizer"):
                # generate an op which trains the model
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
                #self.train_op  = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            # set up tensorflow session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # Uncomment these to record compute graph:

            self.summaries = tf.summary.merge_all()
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(outdir, "summaries"), self.graph)

    def run_graph(self, outputs, data, tt, options=None, run_metadata=None):
        with self.graph.as_default():
            dropout_keep = 1.0
            if tt == "train" and "dropout_keep_prob" in self.layer_args:
                dropout_keep = self.layer_args["dropout_keep_prob"]

            feed_dict = {
                self.in_vertex_r: data["r_vertex"], self.in_edge_r: data["r_edge"],
                self.in_vertex_l: data["l_vertex"], self.in_edge_l: data["l_edge"],
                self.in_hood_indices_r: data["r_hood_indices"],
                self.in_hood_indices_l: data["l_hood_indices"],
                self.examples: data["label"][:, :2],
                self.labels: data["label"][:, 2],
                self.dropout_keep_prob: dropout_keep}

            return self.sess.run(outputs, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    def get_labels(self, data):
        return {"label": data["label"][:, 2, np.newaxis]}

    def predict(self, data):
        logging.debug("|| PWClassifier.predict || runnining predict step")
        loss, preds = self.run_graph([self.loss, self.preds], data, "test")
        results = {"label": preds, "loss": loss}
        return results

    def loss(self, data):
        return self.run_graph(self.loss, data, "test")

    def train(self, data):
        logging.debug("|| PWClassifier.train || running training step")
        _, loss, train_summaries, _, step  = self.run_graph([self.train_op, self.loss, self.summaries, self.inc, self.global_step], data, "train")
        self.train_summary_writer.add_summary(train_summaries, step)
        return loss

    def get_nodes(self):
        return [n for n in self.graph.as_graph_def().node]

    def close(self):
        with self.graph.as_default():
            self.sess.close()
