#!/usr/bin/env python3
# This is an adaptation of an online tutorial
# on Generative Adversarial Networks to test 
# that the customized construction constructs 
# correctly
# Original code: https://github.com/wiseodd/generative-models
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

from ganrecs.network import gan

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def get_one_hot(value):
    zeros = [0 for _ in range(10)]
    zeros[value] = 1
    return zeros

def main():
    print("Constructing network...")
    dis_arch = [784, 128, 1]
    gen_arch = [100, 128, 784]
    network = gan(dis_arch, gen_arch, 10)

    print("Reading input data...")
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print("Checking out directory...")
    if not os.path.exists('out/'):
        os.makedirs('out/')

    print("Starting run...")
    i = 0
    for it in range(500000):

        X_mb, digits = mnist.train.next_batch(128)
        _sample = sample_Z(128, 100)
        _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss], feed_dict={network.discriminator_input: X_mb, network.generator_input: _sample, network.generator_condition: digits})
        _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss], feed_dict={network.generator_input: _sample, network.generator_condition: digits})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
    
    while True:
        val = int(input("Input a number 0-9 to generate: "))
        val = get_one_hot(val)
        _sample = sample_Z(1,100)
        result = session.run(network.generator.prob, feed_dict={network.generator_input: _sample, network.generator_condition: [val]})
        fig = plot(result)
        fig.show()

if __name__ == '__main__':
    main()
