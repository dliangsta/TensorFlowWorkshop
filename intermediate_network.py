# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

import tensorflow as tf


def construct_intermediate_network():
    # Create the model

    # Placeholder for input data
    x = tf.placeholder(tf.float32, [None, 784])

    # 1st layer's weights, bias, output
    W1 = tf.Variable(tf.random_normal([784, 784], stddev=.1))
    b1 = tf.Variable(tf.random_normal([784], stddev=.1))
    l1 = tf.sigmoid(tf.matmul(x, W1) + b1)

    # 2nd layer's weights, bias, output
    W2 = tf.Variable(tf.random_normal([784, 10], stddev=.1))
    b2 = tf.Variable(tf.random_normal([10], stddev=.1))
    y = l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)

    # Placeholder for real labels
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Loss
    loss = tf.losses.mean_squared_error(labels=y_, predictions=y)

    # Optimizer
    optimizer_op = tf.train.AdamOptimizer(
        learning_rate=1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_, optimizer_op, accuracy_op
