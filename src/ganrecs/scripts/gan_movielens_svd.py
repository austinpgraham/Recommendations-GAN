#!/usr/bin/env python3
import os
import json
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from random import randint
from random import sample

from ganrecs.data import RatingCollection

from ganrecs.network import gan
from ganrecs.network import dragan

from surprise import Dataset

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

MOVIES_COUNT = 3706 # 1682
USER_COUNT = 6040
TEST_PERCENT = .2
DEFAULT_EPOCHS = 50000


def sample_Z(m, n):
    return np.random.normal(-1., 1., size=[m, n])


def process_args(args=None):
    parser = argparse.ArgumentParser(description="Test with MNIST data set")
    parser.add_argument('-n', '--noise', help="Amount of noise to include in network")
    parser.add_argument('-e', '--epochs', help="Number of epochs to run")
    parser.add_argument('-f', '--file', help="Output file name")
    parser.add_argument('-p', '--pca', help="Number of latent SVD features.")
    args = parser.parse_args(args)

    assert args.file, "Must provide output file"
    assert args.noise, "Must provide noise"
    epochs = DEFAULT_EPOCHS if args.epochs is None else int(args.epochs)

    return args.file, int(args.noise), epochs, int(args.pca)


def get_sample(data, size):
    user_keys = list(data.keys())
    indices = sample(user_keys, size)
    result = []
    for i in indices:
        result.append(list(data[str(i)].values()))
    return np.array(result)


def plot_losses(epochs, d_losses, g_losses):
    xs = [x for x in range(epochs)]
    plt.title('D/G Losses Over Time')
    plt.plot(xs, d_losses, label='Discriminator')
    plt.plot(xs, g_losses, label='Generator')
    plt.legend()
    plt.show()

def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

def write_output(d_losses, g_losses, distances, _file):
    d_losses = [float(d) for d in d_losses]
    g_losses = [float(g) for g in g_losses]
    distances = [float(d) for d in distances]
    _dict = {'d_losses': d_losses, 'g_losses': g_losses, 'distances': distances}
    json.dump(_dict, open('permout/{}'.format(_file), 'w'))

def main(args=None):
    _file, noise, epochs, pca_com = process_args(args)
    data = Dataset.load_builtin('ml-1m')
    rc = RatingCollection(data.raw_ratings)
    
    print("Constructing network...")

    d_losses = []
    g_losses = []
    print("Starting run...")
    distances = []
    for i in range(len(rc.folds)):
        print("Fold {}...".format(i + 1))
        training_data = {}
        for idx, value in enumerate(rc.folds):
            #if idx != i:
            training_data = {**training_data, **rc._get_matrix(value)}
        print('Calculating principle components...')
        pca = PCA(pca_com)
        pca.fit(get_sample(training_data, len(training_data.keys())))
        dis_arch = [MOVIES_COUNT, 300, 1]
        gen_arch = [noise, 300, MOVIES_COUNT]
        tf.reset_default_graph()
        network = gan(dis_arch, gen_arch, pca_com, 50)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        for it in range(epochs):
            users = get_sample(training_data, 50)
            _sample = sample_Z(50, noise)
            users_p = get_perturbed_batch(users)
            users_pca = pca.transform(users)
            # _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss],
            # feed_dict={network.discriminator_input: users, network.generator_input: _sample,
            # network.generator_condition: users_pca, network.pert: users_p, network.keep_prob: 0.5})
            _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss],
            feed_dict={network.discriminator_input: users, network.generator_input: _sample,
            network.generator_condition: users_pca, network.keep_prob: 0.5})
            _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss],
            feed_dict={network.generator_input: _sample, network.generator_condition: users_pca,
                        network.keep_prob: 0.5})

            if it % 100 == 0:
                d_losses.append(D_loss_curr)
                g_losses.append(G_loss_curr)
                print('Iteration {} of {} ---------------'.format(it, epochs))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))

                # Get the classification distances
                test_fold = rc._get_matrix(rc.folds[i])
                sample_size = len(test_fold)
                users = get_sample(test_fold, sample_size).astype(np.float32)
                _sample = sample_Z(sample_size, noise)
                users_pca = pca.transform(users)
                generated_images = session.run(network.generator.prob, feed_dict={network.generator_input: _sample,             
                network.generator_condition: users_pca})

                feed_users = get_sample(test_fold, sample_size).astype(np.float32)
                feed_users = tf.convert_to_tensor(feed_users, dtype=tf.float32)
                generated_images = tf.convert_to_tensor(generated_images, dtype=tf.float32)
                result = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(feed_users, generated_images)
                result = session.run(result)
                distances.append(result)

        write_output(d_losses, g_losses, distances, _file)
        break


if __name__ == '__main__':
    main(args)