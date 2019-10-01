# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
tf.enable_eager_execution()
tf.executing_eagerly()

## Permuted MNIST

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(num_tasks_to_run):
	task_permutation.append( np.random.permutation(784) )
  
num_tasks_to_run = 10

num_epochs_per_task = 20

minibatch_size = 32
learning_rate = 0.001


#Based on tutorial provided create your MLP model for above problem
#For TF2.0 users Keras can be used for loading trainable variables and dataset.
#You might need google collab to run large scale experiments
