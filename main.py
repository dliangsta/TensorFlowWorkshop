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

from datetime import datetime
start = datetime.now()


import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from simple_network import construct_simple_network
from intermediate_network import construct_intermediate_network
from advanced_network import construct_advanced_network


FLAGS = None


def main(args):
    if FLAGS.benchmark:
        FLAGS.log_freq = FLAGS.n_itr / 100
        networks = range(1, 4)
    else:
        networks = [FLAGS.network]

    for network in networks:
        # Import data
        mnist = input_data.read_data_sets(
            '/tmp/tensorflow/mnist/input_data', one_hot=True)

        # Get constructor
        if network == 1:
            network_constructor = construct_simple_network
            network_name = 'simple'
        elif network == 2:
            network_constructor = construct_intermediate_network
            network_name = 'intermediate'
        elif network == 3:
            network_constructor = construct_advanced_network
            network_name = 'advanced'
        else:
            raise 'Invalid network specified.'

        print('\nNeural network: %s' % network)

        train(mnist, network_name, network_constructor)


def train(data, network_name, network_constructor):
    # Get required ops
    inputs, labels, optimizer_op, accuracy_op = network_constructor()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train
        for i in range(FLAGS.n_itr + 1):
            batch_xs, batch_ys = data.train.next_batch(FLAGS.batch_size)
            sess.run(optimizer_op, feed_dict={
                     inputs: batch_xs, labels: batch_ys})

            # Test trained model every 10 iterations
            if not i % FLAGS.log_freq:
                accuracy = sess.run(accuracy_op, feed_dict={
                                    inputs: data.test.images, labels: data.test.labels})
                if i == 0:
                    print('\n\t     t, accuracy')
                print('\t%6d, %.9f' % (i, accuracy))

        print('\nNeural network: %s' % network_name)
        print('Final accuracy: %.9f' % accuracy)
        print('Time taken    : %s' % str(datetime.now() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=int, default=1,
                        help='which neural network to use: \'1\':simple \'2\':intermediate or \'3\':advanced cnn')
    parser.add_argument('--n_itr', type=int, default=int(1e4),
                        help='number of iterations')
    parser.add_argument('--log_freq', type=int, default=int(1e3),
                        help='number of iterations to skip after logging')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of tuples per iteration to feed')
    parser.add_argument('--benchmark', type=bool, default=False,
                        help='benchmark by running all networks')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
