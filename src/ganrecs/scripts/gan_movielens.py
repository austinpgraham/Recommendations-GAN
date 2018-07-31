#!/usr/bin/env python3
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from random import randint
from random import sample

from ganrecs.data import RatingCollection

from ganrecs.network import gan

from surprise import Dataset

import matplotlib.pyplot as plt

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

MOVIES_COUNT = 3706
USER_COUNT = 6040
TEST_PERCENT = .2


def sample_Z(m, n):
    return np.random.normal(-1., 1., size=[m, n])


def process_args(args=None):
    parser = argparse.ArgumentParser(description="Test with MNIST data set")
    parser.add_argument('-l', '--location', help='Saved model location')
    parser.add_argument('-n', '--noise', help="Amount of noise to include in network")
    parser.add_argument('-e', '--epochs', help="Number of epochs to run")
    args = parser.parse_args(args)

    assert args.location, "Must provide output location"
    assert args.noise, "Must provide noise"
    assert args.epochs, "Must provide epochs"

    location = os.path.expanduser(args.location)

    if not os.path.exists(location):
        os.makedirs(location)
    else:
        ans = input("Overwrite output directory?: ").upper()
        if ans == 'N' or ans == 'NO':
            print('Exiting...')
            exit()
        shutil.rmtree(location)

    return location, int(args.noise), int(args.epochs)


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


def main(args=None):
    location, noise, epochs = process_args(args)
    model_path = os.path.join(location, "model.ckpt")
    data = Dataset.load_builtin('ml-1m')
    rc = RatingCollection(data.raw_ratings)

    print("Constructing network...")

    d_losses = []
    g_losses = []
    if os.path.exists(model_path + ".meta"):
        print("Restoring model....")
        saver.restore(session, model_path)
    else:
        print("Starting run...")
        distances = []
        for i in range(len(rc.folds)):
            training_data = {}
            for idx, value in enumerate(rc.folds):
                if idx != i:
                    training_data = {**training_data, **rc._get_matrix(value)}
            dis_arch = [MOVIES_COUNT, 2000, 1000, 1]
            gen_arch = [noise, 3000, 2000, 3000, MOVIES_COUNT]
            tf.reset_default_graph()
            network = gan(dis_arch, gen_arch, MOVIES_COUNT)
            session = tf.Session()
            session.run(tf.global_variables_initializer())
            for it in range(epochs):
                users = get_sample(training_data, 50)
                _sample = sample_Z(50, noise)
                _, D_loss_curr = session.run([network.discriminator_optimizer, network.discriminator_loss],
                feed_dict={network.discriminator_input: users, network.generator_input: _sample,
                network.generator_condition: users})
                _, G_loss_curr = session.run([network.generator_optimizer, network.generator_loss],
                feed_dict={network.generator_input: _sample, network.generator_condition: users})

                if it % 100 == 0:
                    d_losses.append(D_loss_curr)
                    g_losses.append(G_loss_curr)
                    print('Iter: {}'.format(it))
                    print('D loss: {:.4}'.format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    d_losses.append(D_loss_curr)
                    g_losses.append(G_loss_curr)
                    print()

            # Get the classification distances
            test_fold = rc._get_matrix(rc.folds[i])
            sample_size = len(test_fold)
            users = get_sample(test_fold, sample_size).astype(np.float32)
            _sample = sample_Z(sample_size, noise)
            generated_images = session.run(network.generator.prob, feed_dict={network.generator_input: _sample,             
            network.generator_condition: users})

            feed_users = get_sample(test_fold, sample_size).astype(np.float32)
            feed_users = tf.convert_to_tensor(feed_users, dtype=tf.float32)
            generated_images = tf.convert_to_tensor(generated_images, dtype=tf.float32)
            result = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(feed_users, generated_images)
            result = session.run(result)
            distances.append(result)
            session.close()
        
        xs = [x for x in range(len(rc.folds))]
        plt.title('Distances of folds')
        plt.plot(xs, distances, label='Frechet Classifier Distance')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main(args)