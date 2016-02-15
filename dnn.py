#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Deep greedy layerwise MLP
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.

# shamelessly taken and basically completely modified by Howon Lee in 2016, cuz BSD allows that
# only Howon doesn't know which version of the BSD license Howon should replicate
# so whatever version NP Rougier wanted, consider it replicated
# dont think that NP Rougier has heard of Howon, because he prolly hasn't
# nor vice versa, now that Howon thinks about it
# -----------------------------------------------------------------------------

# modified by Howon Lee to make a point about the fractal nature of backprop
# play with the params as you like
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as npr
import scipy.sparse as sci_sp
import matplotlib.pyplot as plt
import cPickle
import collections
import time
import sys
import math
import random
import gzip

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

def mat_dsigmoid(mat):
    new_mat = mat.copy()
    new_mat.data = 1.0 - (new_mat.data ** 2)
    return new_mat

def sparse_outer(fst, snd, to_calc):
    new = np.zeros(to_calc.shape)
    fst = fst.toarray()
    snd = snd.toarray()
    for row_idx, col_idx in zip(to_calc.row, to_calc.col):
        # matrices, remember
        new[row_idx, col_idx] = fst[0, row_idx] * snd[0, col_idx]
    return sci_sp.csr_matrix(new)

class MLP:
    '''
    This is used via SGD only in the MNIST thing Howon rigged up
    '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)
        self.bp_times = []

        # Build layers
        self.layers = []
        # Input layer (add bias in the data stage)
        self.layers.append(sci_sp.csc_matrix(np.ones(self.shape[0])))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(sci_sp.csc_matrix(np.ones(self.shape[i])))

        # Build weights matrix (randomly)
        self.weights = []
        self.sparsifiers = []
        self.has_sparsified = False
        for i in range(n-1):
            # new_weights = (2 * (npr.random((self.layers[i].size, self.layers[i+1].size))) - 1) * math.pow(math.e, -(n-i-1))
            new_weights = (2 * npr.random((self.layers[i].size, self.layers[i+1].size)) - 1)
            new_weights *= np.sqrt(6)
            new_weights /= np.sqrt(self.layers[i].size + self.layers[i+1].size)
            print >> sys.stderr, str(i) + " initialized"
            print >> sys.stderr, "max on this layer: ", np.max(new_weights)
            self.weights.append(sci_sp.csc_matrix(new_weights))
            self.sparsifiers.append(sci_sp.coo_matrix(np.ones_like(new_weights)))

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''
        ''' Data is still in numpy format, hear? '''

        # Set input layer
        self.layers[0][0, :] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(self.layers[i-1].dot(self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.001):
        ''' Back propagate error related to target using lrate. '''
        begin_time = time.clock()

        deltas = []

        # Compute error on output layer
        error = sci_sp.csc_matrix(target - self.layers[-1])
        delta = error.multiply(mat_dsigmoid(self.layers[-1]))
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            error = sci_sp.csc_matrix(deltas[0].dot(self.weights[i].T))
            delta = error.multiply(mat_dsigmoid(self.layers[i]))
            deltas.insert(0,delta)

        # Update weights: this is the bit that scales
        for i in range(len(self.weights)):
            if i < len(self.weights)-1:
                dw = sparse_outer(self.layers[i], deltas[i], self.sparsifiers[i])
            else:
                dw = self.layers[i].T.dot(deltas[i]) ### dense at the last one
            self.weights[i] += lrate*dw

        # Return error
        end_time = time.clock()
        self.bp_times.append(end_time - begin_time)
        return error.sum()

    def check_sparsity(self):
        for i in range(len(self.weights)):
            print >> sys.stderr, len(self.weights[i].indices), " / ",\
                    reduce(lambda x, y: x * y, self.weights[i].shape)
            print >> sys.stderr, np.mean(np.abs(self.weights[i].toarray().ravel()))

    def random_sparsify(self, sparsity_percentage):
        # experimentation
        self.has_sparsified = True
        chance_to_remain = (100 - sparsity_percentage) / float(100)
        for i in range(len(self.weights)-1): # not the softmax layer
            new_sparsifier = self.sparsifiers[i].toarray()
            self.sparsifiers[i] = sci_sp.coo_matrix(npr.random(size=self.weights[i].shape) < chance_to_remain, new_sparsifier)
            # self.weights[i][self.sparsifiers[i]] = 0
            # self.weights[i].eliminate_zeros()

    def sparsify(self, sparsity_percentage):
        self.has_sparsified = True
        for i in range(len(self.weights)-1):
            all_weights = np.abs(self.weights[i].toarray().ravel())
            pos_weights = all_weights[all_weights > 0]
            thresh = np.percentile(pos_weights, sparsity_percentage)
            new_sparsifier = self.sparsifiers[i].toarray()
            self.sparsifiers[i] = sci_sp.coo_matrix(np.logical_and(np.abs(self.weights[i].toarray()) > thresh, new_sparsifier))
            self.weights[i][np.abs(self.weights[i].toarray()) < thresh] = 0
            self.weights[i].eliminate_zeros()

    def first_sparsify(self, sparsity_percentage):
        self.has_sparsified = True
        all_weights = np.abs(self.weights[0].toarray().ravel())
        pos_weights = all_weights[all_weights > 0]
        thresh = np.percentile(pos_weights, sparsity_percentage)
        new_sparsifier = self.sparsifiers[0].toarray()
        self.sparsifiers[0] = sci_sp.coo_matrix(np.logical_and(np.abs(self.weights[0].toarray()) > thresh, new_sparsifier))
        self.weights[0][np.abs(self.weights[0].toarray()) < thresh] = 0
        self.weights[0].eliminate_zeros()

    def total_sparsify(self, sparsity_percentage):
        self.has_sparsified = True
        all_weights = np.abs(np.hstack(tuple([weight.toarray().ravel() for weight in self.weights])))
        pos_weights = all_weights[all_weights > 0]
        thresh = np.percentile(pos_weights, sparsity_percentage)
        for i in range(len(self.weights)-1):
            new_sparsifier = self.sparsifiers[i].toarray()
            self.sparsifiers[i] = sci_sp.coo_matrix(np.logical_and(np.abs(self.weights[i].toarray()) > thresh, new_sparsifier))
            self.weights[i][np.abs(self.weights[i].toarray()) < thresh] = 0
            self.weights[i].eliminate_zeros()

    def layer_sparsify(self, sparsity_percentages):
        self.has_sparsified = True
        for i in range(len(self.weights)-1):
            all_weights = np.abs(self.weights[i].toarray().ravel())
            pos_weights = all_weights[all_weights > 0]
            thresh = np.percentile(pos_weights, sparsity_percentages[i])
            new_sparsifier = self.sparsifiers[i].toarray()
            self.sparsifiers[i] = sci_sp.coo_matrix(np.logical_and(np.abs(self.weights[i].toarray()) > thresh, new_sparsifier))
            self.weights[i][np.abs(self.weights[i].toarray()) < thresh] = 0
            self.weights[i].eliminate_zeros()

def onehots(n):
    arr = np.array([-1.0] * 10)
    arr[n] = 1.0
    return arr

def create_mnist_samples(filename="mnist.pkl.gz"):
    samples = np.zeros(50000, dtype=[('input',  float, 785), ('output', float, 10)])
    with gzip.open(filename, "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)
        for x in xrange(50000):
            train_point = np.ones(785)
            train_point[0:784] = train_set[0][x]
            samples[x] = train_point, onehots(train_set[1][x])
    return samples, 784

def create_cifar_samples(filename="cifar-10-batches-py/data_batch_1"):
#### gotta add biases!
    samples = np.zeros(10000, dtype=[('input',  float, 3072), ('output', float, 10)])
    with open(filename, "rb") as f:
        cifar_dict = cPickle.load(f)
        for x in xrange(10000):
            # CIFAR is uint8s, but I would like floats
            samples[x] = cifar_dict["data"][x] / 256.0, onehots(cifar_dict["labels"][x])
    return samples, 3072

def test_network(net, samples):
    correct, total = 0, 0
    for x in xrange(samples.shape[0]):
        total += 1
        in_pat = samples["input"][x]
        out_pat = samples["output"][x]
        # needs to be adjusted for neural net's sparsity datastruct
        out = net.propagate_forward(in_pat).toarray().ravel()
        if np.argmax(out) == np.argmax(out_pat):
            correct += 1
    # lots of less naive things out there
    return float(correct) / float(total)

def test_sparsify(num_epochs, sparsity_percentages, num_burnin, num_iters, architecture):
    total_begin_time = time.clock()
    samples, dims = create_mnist_samples()
    network = MLP(*architecture)
    burnin_iter = 0
    while test_network(network, samples[40500:40520]) <= num_burnin:
        for x in xrange(50):
            burnin_iter += 1
            n = np.random.randint(40000)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n], lrate=0.05)
        print >> sys.stderr, "burnin: ", burnin_iter, time.clock(), test_network(network, samples[40500:40520])
    print >> sys.stderr, "burnin finished"
    network.check_sparsity()
    network.layer_sparsify(sparsity_percentages)
    prev_time = time.clock()
    for epoch in xrange(num_epochs):
        for i in xrange(num_iters):
            if i % 20 == 0:
                print >> sys.stderr, "==============="
                print >> sys.stderr, "sample: ", i, " / ", num_iters, " time: ", time.clock()
                print >> sys.stderr, "epoch: ", epoch, " time taken: ", time.clock() - prev_time
                if network.bp_times:
                    print >> sys.stderr, "last bp_time: ", network.bp_times[-1]
                prev_time = time.clock()
                network.check_sparsity()
                print >> sys.stderr, test_network(network, samples[40500:40520])
                print >> sys.stderr, "==============="
            network.propagate_forward(samples['input'][i])
            network.propagate_backward(samples['output'][i])
        # was also expanding, but that doesn't work as well
        network.check_sparsity()
    print "test: ", test_network(network, samples[40000:40500])

def make_hiddens(network, prev_hidden):
    network_layer_size = network.layers[-2].size
    new_hiddens = np.zeros(50000, dtype=[('input',  float, network_layer_size + 1), ('output', float, 10)]) # +1 for bias
    for i in xrange(prev_hidden.shape[0]):
        if i % 2000 == 0:
            print "hidden creation : ", i, " / ", prev_hidden.shape[0]
        network.propagate_forward(prev_hidden['input'][i])
        curr_new_hidden = np.ones(network_layer_size + 1) #+1 for bias
        curr_new_hidden[0:-1] = network.layers[-2].toarray().ravel()
        new_hiddens[i] = curr_new_hidden, prev_hidden['output'][i]
    return new_hiddens

def smash_networks(network_list):
    new_architecture = []
    new_architecture.append(network_list[0].layers[0].size)
    for network in network_list:
        new_architecture.append(network.layers[1].size + 1)
    new_architecture.append(network_list[-1].layers[-1].size)
    print new_architecture
    total_net = MLP(*new_architecture)
    for idx, network in enumerate(network_list):
        new_weights = np.zeros((network.layers[0].size, network.layers[1].size + 1))
        # just copying is bugging out on me?
        for x in xrange(network.weights[0].shape[0]):
            for y in xrange(network.weights[0].shape[1]):
                new_weights[x,y] = network.weights[0][x,y]
        total_net.weights[idx] = new_weights
        print new_weights.shape
    last_net = network_list[-1]
    last_weights = np.zeros((last_net.layers[1].size + 1, last_net.layers[2].size))
    print last_weights.shape
    for x in xrange(last_net.weights[-1].shape[0]):
        for y in xrange(last_net.weights[-1].shape[1]):
            last_weights[x,y] = last_net.weights[-1][x,y]
    total_net.weights[-1] = last_weights
    return total_net

def test_deep_layerwise_sparse(num_layers, sparsity_percentages, num_burnin, num_iters, num_hiddens):
    # this has a different attitude towards "layers"
    total_begin_time = time.clock()
    samples, dims = create_mnist_samples()
    networks = []
    tests = []
    previous_hiddens = [samples]
    for network_idx in range(num_layers):
        curr_hiddens = previous_hiddens[-1]
        curr_network = MLP(curr_hiddens['input'][0].size, num_hiddens, curr_hiddens['output'][0].size)
        burnin_iter = 0
        while test_network(curr_network, curr_hiddens[40000:40020]) <= num_burnin:
            for x in xrange(50):
                burnin_iter += 1
                n = np.random.randint(40000)
                curr_network.propagate_forward(curr_hiddens['input'][n])
                curr_network.propagate_backward(curr_hiddens['output'][n], lrate=0.05)
            print >> sys.stderr, "burnin: ", burnin_iter, time.clock(), test_network(curr_network, curr_hiddens[40000:40020])
        print >> sys.stderr, "burnin finished"
        curr_network.layer_sparsify(sparsity_percentages)
        prev_time = time.clock()
        for i in xrange(num_iters):
            if i % 100 == 0:
                print >> sys.stderr, "==============="
                print >> sys.stderr, "sample: ", i, " / ", num_iters, " time: ", time.clock()
                print >> sys.stderr, "time taken: ", time.clock() - prev_time
                print >> sys.stderr, "network: ", network_idx
                if curr_network.bp_times:
                    print >> sys.stderr, "last bp_time: ", curr_network.bp_times[-1]
                prev_time = time.clock()
                curr_network.check_sparsity()
                print >> sys.stderr, test_network(curr_network, curr_hiddens[40500:40520])
                print >> sys.stderr, "==============="
            curr_network.propagate_forward(curr_hiddens['input'][i])
            curr_network.propagate_backward(curr_hiddens['output'][i])
        previous_hiddens.append(make_hiddens(curr_network, curr_hiddens))
        networks.append(curr_network)
        test_val = test_network(curr_network, curr_hiddens[40020:45000])
        print "test: ", test_val
        tests.append(test_val)
    final_network = smash_networks(networks)
    print "test: ", test_network(final_network, samples[40020:45000])
    print "all tests: ", tests

if __name__ == '__main__':
    num_hiddens = 200
    sparsities = [0]
    test_deep_layerwise_sparse(num_layers=10, sparsity_percentages=sparsities, num_burnin=0.0, num_iters=10000, num_hiddens=num_hiddens)
