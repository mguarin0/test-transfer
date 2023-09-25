import numpy as np
import tensorflow as tf
import logging

__all__ = [
    "no_conv",
    "diffusion_convolution",
    "single_weight_matrix",
    "node_average",
    "node_edge_average",
    "order_dependent",
    "deep_tensor_conv",
    "dense", 
    "merge",
    "average_predictions",
    "initializer",
    "nonlinearity",
]

""" ====== Layers ====== """
""" All layers have as first two parameters:
        - input: input tensor or tuple of input tensors
        - params: dictionary of parameters, could be None
    and return tuple containing:
        - output: output tensor or tuple of output tensors
        - params: dictionary of parameters, could be None
"""


def no_conv(input, params, leg_type, experiment_name, replica_number, layer, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, _, _ = input
    vertices = tf.nn.dropout(vertices, dropout_keep_prob)
    v_shape = vertices.get_shape()
    if params is None:
        # create new weights
        Wvc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvc_{}".format(leg_type), trainable=trainable)  # (v_dims, filters)
        bv = tf.Variable(initializer("zero", (filters,)), name="bv_{}".format(leg_type), trainable=trainable)
    else:
        # use shared weights
        Wvc = params["Wvc_{}".format(leg_type)]
        bv = params["bv_{}".format(leg_type)]
        filters = Wvc.get_shape()[-1].value
    params = {"Wvc_{}".format(leg_type): Wvc, "bv_{}".format(leg_type): bv}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wvc, name="Zc_{}".format(leg_type))  # (n_verts, filters)
    nonlin = nonlinearity("relu")
    sig = Zc + bv
    z = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, dropout_keep_prob)

    """
    try:
        tf.summary.histogram("histogram_Wvc/{}/{}/no_conv_{}".format(experiment_name, replica_number, layer), Wvc)
        tf.summary.histogram("histogram_bv/{}/{}/no_conv_{}".format(experiment_name, replica_number, layer), bv)
    except Exception as err:
        pass
    """ 
    return z, params


def node_average(input, params, leg_type, experiment_name, replica_number, layer, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value

    if params is None:
        # create new weights
        Wc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wc_{}".format(leg_type), trainable=trainable)  # (v_dims, filters)
        Wn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wn_{}".format(leg_type), trainable=trainable)  # (v_dims, filters)
        b = tf.Variable(initializer("zero", (filters,)), name="b_{}".format(leg_type), trainable=trainable)
    else:
        Wn, Wc = params["Wn_{}".format(leg_type)], params["Wc_{}".format(leg_type)]
        filters = Wc.get_shape()[-1].value
        b = params["b_{}".format(leg_type)]
    params = {"Wn_{}".format(leg_type): Wn, "Wc_{}".format(leg_type): Wc, "b_{}".format(leg_type): b}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wc, name="Zc_{}".format(leg_type))  # (n_verts, filters)
    # create neighbor signals
    v_Wn = tf.matmul(vertices, Wn, name="v_Wn_{}".format(leg_type))  # (n_verts, filters)
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wn, nh_indices), 1),
                   tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)

    nonlin = nonlinearity("relu")
    sig = Zn + Zc + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    #h = tf.nn.dropout(h, dropout_keep_prob)

    """
    try:
        tf.summary.histogram("histogram_Wc/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), Wc)
        tf.summary.histogram("histogram_Wn/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), Wn)
        tf.summary.histogram("histogram_b/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), b)
    except Exception as err:
        pass
    """

    return h, params


def node_edge_average(input, params, leg_type, experiment_name, replica_number, layer, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    # TODO make sure that these variables are not shared across r and l
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2) # shape (None, 20, 1) -> (None, 20)
    v_shape = vertices.get_shape() # shape (None, 70)
    e_shape = edges.get_shape() # shape (None, 20, 2)
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value
    # nh_sizes has shape (None, 1)
    if params is None:
        # create new weights
        Wvc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvc_{}".format(leg_type), trainable=trainable)  # (70, filters)
        bv = tf.Variable(initializer("zero", (filters,)), name="bv_{}".format(leg_type), trainable=trainable) # (filters, )
        Wvn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvn_{}".format(leg_type), trainable=trainable)  # (70, filters)
        We = tf.Variable(initializer("he", (e_shape[2].value, filters)), name="We_{}".format(leg_type), trainable=trainable)  # (2, filters)
    else:
        # use shared weights
        Wvn, We = params["Wvn_{}".format(leg_type)], params["We_{}".format(leg_type)]
        Wvc = params["Wvc_{}".format(leg_type)]
        bv = params["bv_{}".format(leg_type)]
        filters = Wvc.get_shape()[-1].value
    params = {"Wvn_{}".format(leg_type): Wvn,
              "We_{}".format(leg_type): We,
              "Wvc_{}".format(leg_type): Wvc,
              "bv_{}".format(leg_type): bv}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wvc, name="Zc_{}".format(leg_type))  # (None, filters)
    # create neighbor signals
    e_We = tf.tensordot(edges, We, axes=[[2], [0]], name="e_We_{}".format(leg_type))  # (None, 20, filters)
    v_Wvn = tf.matmul(vertices, Wvn, name="v_Wvn_{}".format(leg_type))  # (None, filters)

    # tf.gather(v_Wvn, nh_indices): (None, 20, 256)
    # tf.reduce_sum(tf.gather(v_Wvn, nh_indices), 1): (None, 256)
    # tf.reduce_sum(e_We, 1): (None, )
    # tf.maximum(nh_sizes, tf.ones_like(nh_sizes)): (None, 1)
    # tf.add(tf.reduce_sum(tf.gather(v_Wvn, nh_indices), 1), tf.reduce_sum(e_We, 1)))
    Zn = tf.divide(tf.add(tf.reduce_sum(tf.gather(v_Wvn, nh_indices), 1), tf.reduce_sum(e_We, 1)),
                   tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (None, )
    nonlin = nonlinearity("relu")
    sig = Zn + Zc + bv
    z = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, dropout_keep_prob)
    
    """
    try:
        tf.summary.histogram("histogram_Wvc/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), Wvc)
        tf.summary.histogram("histogram_bv/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), bv)
        tf.summary.histogram("histogram_Wvn/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), Wvn)
        tf.summary.histogram("histogram_We/{}/{}/msgLayer_{}".format(experiment_name, replica_number, layer), We)
    except Exception as err:
        pass
    """
    return z, params


def dense(input, params, experiment_name, replica_number, layer, out_dims=None, dropout_keep_prob=1.0, nonlin=True, trainable=True, **kwargs):
    input = tf.nn.dropout(input, dropout_keep_prob)
    in_dims = input.get_shape()[-1].value
    out_dims = in_dims if out_dims is None else out_dims
    if params is None:
        W = tf.Variable(initializer("he", [in_dims, out_dims]), name="w", trainable=trainable)
        b = tf.Variable(initializer("zero", [out_dims]), name="b", trainable=trainable)
        params = {"W": W, "b": b}
    else:
        W, b = params["W"], params["b"]
    Z = tf.matmul(input, W) + b
    if(nonlin):
        nonlin = nonlinearity("relu")
        act = nonlin(Z)
        act = tf.nn.dropout(act, dropout_keep_prob)
    else:
        act = tf.nn.dropout(Z, dropout_keep_prob)

    """
    try:
        tf.summary.histogram("histogram_W/{}/{}/denseLayer_{}".format(experiment_name, replica_number, layer), W)
        tf.summary.histogram("histogram_b/{}/{}_denseLayer_{}".format(experiment_name, replica_number, layer), b)
    except Exception as err:
        pass
    """ 
    return act, params


def merge(input, _, **kwargs):
    input_r, input_l, examples = input
    out_r = tf.gather(input_r, examples[:, 1])
    out_l = tf.gather(input_l, examples[:, 0])
    output = tf.concat([out_r, out_l], axis=1)
    #output_l = tf.concat([out_l, out_r], axis=0)
    #return tf.concat((output_r, output_l), axis=1), None
    return output, None


def average_predictions(input, _, **kwargs):
    logging.debug("+++++++++++++++++++++++++++++++ average_predictions ++++++++++++++++++++++++")
    combined = tf.reduce_mean(tf.stack(tf.split(input, 2)), 0)
    return combined, None


""" ======== Non Layers ========= """


def initializer(init, shape):
    if init == "zero":
        return tf.zeros(shape)
    elif init == "he":
        fan_in = np.prod(shape[0:-1])
        std = 1/np.sqrt(fan_in)
        return tf.random_uniform(shape, minval=-std, maxval=std)


def nonlinearity(nl):
    if nl == "relu":
        return tf.nn.relu
    elif nl == "tanh":
        return tf.nn.tanh
    elif nl == "linear" or nl == "none":
        return lambda x: x
