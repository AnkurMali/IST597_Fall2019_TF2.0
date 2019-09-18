"""
author:-aam35
"""
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tfe.enable_eager_execution()

# Create data
NUM_EXAMPLES = 500

#define inputs and outputs with some noise 
X = tf.random_normal([NUM_EXAMPLES])  #inputs 
noise = tf.random_normal([NUM_EXAMPLES]) #noise 
y = X * 3 + 2 + noise  #true output

# Create variables.
W = None
b = None


train_steps = 1000
learning_rate = 0.001

# Define the linear predictor.
def prediction(x):
  return None

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
  return None

def huber_loss(y, y_predicted, m=1.0):
  """Huber loss."""
  return None

for i in range(train_steps):
  ###TO DO ## Calculate gradients
plt.plot(X, y, 'bo',label='org')
plt.plot(X, y * W.numpy() + b.numpy(), 'r',
         label="huber regression")
plt.legend()
plt.show
