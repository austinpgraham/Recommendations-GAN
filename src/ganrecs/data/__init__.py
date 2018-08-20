#!/usr/bin/env python3
import json

import numpy as np

from random import sample
from random import shuffle


class Rating():

    def __init__(self, data_tuple):
        self.user, self.item, self.rating, _ = data_tuple
    
    def as_list(self):
        return [self.user, self.item, self.rating]


class RatingCollection():

    SAMPLE_AMOUNT = 0.2
    FOLDS = 10
    FILE = "__tmp__.csv"

    def __init__(self, _file, sep=',', _max=5.):
        self.keys = []
        user_items = []
        start = None
        with open(_file, 'r') as fp:
            line = fp.readline()
            while line:
                user, item, rating = line.split(sep)
                if start is None:
                    start = user
                if user not in self.keys:
                    self.keys.append(user)
                if start != user:
                    with open(self.FILE, 'a') as sfp:
                        sfp.write(",".join(user_items))
                        user_items = []
                        start = user
                user_items.append(str(float(rating) / _max))
                line = fp.readline()


    def get_sample(self, size):
        indices = sample(list(range(len(self.keys))), size)
        _sample = []
        with open(self.FILE, 'r') as fp:
            for i, line in enumerate(fp):
                if i in indices:
                    _sample.append([float(x) for x in line.split(',')])
        return np.array(_sample)
