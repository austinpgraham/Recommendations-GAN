#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
import numpy as np

from random import randint
from random import sample

from ganrecs.data import RatingCollection

from ganrecs.network import gan
from ganrecs.network import dragan

from sklearn.decomposition import PCA

from google.cloud import storage

import tensorflow as tf


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

ARTISTS_COUNT = 98211
USER_COUNT = 1948882
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
    parser.add_argument('-i', '--input', help="Ratings input file location")
    args, _ = parser.parse_known_args(args)

    assert args.file, "Must provide output file"
    assert args.input, "Must provide input file"
    assert args.noise, "Must provide noise"
    epochs = DEFAULT_EPOCHS if args.epochs is None else int(args.epochs)
    _input_file = os.path.expanduser(args.input)

    return args.file, int(args.noise), epochs, int(args.pca), _input_file

def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

def write_output(d_losses, g_losses, distances, _file):
    d_losses = [float(d) for d in d_losses]
    g_losses = [float(g) for g in g_losses]
    distances = [float(d) for d in distances]
    _dict = {'d_losses': d_losses, 'g_losses': g_losses, 'distances': distances}
    json.dump(_dict, open(_file, 'w'))

def read_data(_input_file):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.dirname(os.path.abspath(__file__)) + '/ext_data/GanRecommendations-16ed4b031d72.json'
    client = storage.Client()
    bucket = client.get_bucket('ganrecommendations-mlengine')
    blob = bucket.get_blob('ratings.txt')
    with open(_input_file, 'wb') as fp:
        blob.download_to_file(fp)

def main(args=None):
    _file, noise, epochs, pca_com, _input_file = process_args(args)
    read_data(_input_file)
    rc = RatingCollection(_input_file)
    print("Constructing network...")

    d_losses = []
    g_losses = []
    print("Starting run...")
    distances = []
    print('Calculating principle components...')
    pca = PCA(pca_com)
    pca.fit(rc.get_sample(len(rc.keys)))
    dis_arch = [ARTISTS_COUNT, 300, 1]
    gen_arch = [noise, 300, ARTISTS_COUNT]
    tf.reset_default_graph()
    network = gan(dis_arch, gen_arch, pca_com, 50)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for it in range(epochs):
        users = rc.get_sample(50)
        _sample = sample_Z(50, noise)
        # users_p = get_perturbed_batch(users)
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
            # This is temporary
            sample_size = 100
            users = rc.get_sample(sample_size).astype(np.float32)
            _sample = sample_Z(sample_size, noise)
            users_pca = pca.transform(users)
            generated_images = session.run(network.generator.prob, feed_dict={network.generator_input: _sample,             
            network.generator_condition: users_pca})

            feed_users = rc.get_sample(sample_size).astype(np.float32)
            feed_users = tf.convert_to_tensor(feed_users, dtype=tf.float32)
            generated_images = tf.convert_to_tensor(generated_images, dtype=tf.float32)
            result = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(feed_users, generated_images)
            result = session.run(result)
            distances.append(result)
            if D_loss_curr <= 5e-8:
                print("Exiting early...")
                break


    write_output(d_losses, g_losses, distances, _file)


if __name__ == '__main__':
    main(sys.argv)
    os.remove("__tmp__.csv")
